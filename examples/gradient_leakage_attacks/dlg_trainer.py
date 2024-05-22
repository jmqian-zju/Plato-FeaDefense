import math
import pickle
import random

import numpy as np
import torch
from plato.config import Config
from plato.trainers import basic
from torchvision import transforms

from defense.GradDefense.dataloader import get_root_set_loader
from defense.GradDefense.sensitivity import compute_sens
from defense.Outpost.perturb import compute_risk
from utils.utils import cross_entropy_for_onehot, label_to_onehot

from utils.utils import random_sparsify, random_sparsify_channels, cosine_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# 设置随机种子
# seed = 0
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True


criterion = cross_entropy_for_onehot
tt = transforms.ToPILImage()


class Trainer(basic.Trainer):
    """The federated learning trainer for the gradient leakage attack."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        callbacks: The callbacks that this trainer uses.
        """

        def weights_init(m):
            """Initializing the weights and biases in the model."""
            if hasattr(m, "weight"):
                m.weight.data.uniform_(-0.5, 0.5)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data.uniform_(-0.5, 0.5)

        super().__init__(model=model, callbacks=callbacks)

        # DLG explicit weights initialziation
        if (
            hasattr(Config().algorithm, "init_params")
            and Config().algorithm.init_params
        ):
            self.model.apply(weights_init)

        self.examples = None
        self.trainset = None
        self.full_examples = None
        self.full_labels = None
        self.full_onehot_labels = None
        self.list_grad = None
        self.target_grad = None
        self.feature_fc1_graph = None
        self.fea_graph_list = None
        self.sensitivity = None

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Creates an instance of the trainloader."""
        # Calculate sensitivity with the trainset
        if hasattr(Config().algorithm, "defense"):
            if Config().algorithm.defense == "GradDefense":
                root_set_loader = get_root_set_loader(trainset)
                self.sensitivity = compute_sens(
                    model=self.model.to(self.device),
                    rootset_loader=root_set_loader,
                    device=Config().device(),
                )

        return torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
        )

    def train_run_start(self, config):
        """Method called at the start of training run."""
        self.target_grad = None

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop."""
        # Store data in the first epoch (later epochs will still have the same partitioned data)
        if self.current_epoch == 1:
            try:
                self.full_examples = torch.cat((examples, self.full_examples), dim=0)
                self.full_labels = torch.cat((labels, self.full_labels), dim=0)
            except:
                self.full_examples = examples
                self.full_labels = labels

            self.full_onehot_labels = label_to_onehot(
                self.full_labels, num_classes=Config().parameters.model.num_classes
            )
        # 新增代码
        # print(examples)
        # print(examples.shape)

        examples.requires_grad = True
        self.examples = examples
        self.model.zero_grad()

        if (
            hasattr(Config().algorithm, "target_eval")
            and Config().algorithm.target_eval
        ):
            # Set model into evaluation mode at client's training
            self.model.eval()
        else:
            self.model.train()

        # Compute gradients in the current step
        if (
            hasattr(Config().algorithm, "defense")
            and Config().algorithm.defense == "GradDefense"
            and hasattr(Config().algorithm, "clip")
            and Config().algorithm.clip is True
        ):
            self.list_grad = []
            for example, label in zip(examples, labels):
                outputs, _ = self.model(torch.unsqueeze(example, dim=0))

                loss = self._loss_criterion(outputs, torch.unsqueeze(label, dim=0))
                grad = torch.autograd.grad(
                    loss,
                    self.model.parameters(),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                )
                self.list_grad.append(list((_.detach().clone() for _ in grad)))
        else:
            try:
                # 新增代码
                if ( hasattr(Config().algorithm, "defense")
                    and Config().algorithm.defense == "MaskDefense"
                ):
                    mask_examples = examples.detach().clone()
                    for i, mask_example in enumerate(mask_examples):
                        mask = random_sparsify(mask_example, Config().algorithm.sparsity)
                        mask_examples[i] = mask_examples[i] * mask
                        # plt.imshow(TF.to_pil_image(mask_examples[i].cpu()))
                        # plt.show()
                    outputs, self.feature_fc1_graph, self.fea_graph_list = self.model(mask_examples)
                else:
                    outputs, self.feature_fc1_graph, self.fea_graph_list = self.model(examples)
            except:
                # 新增代码
                if ( hasattr(Config().algorithm, "defense")
                    and Config().algorithm.defense == "MaskDefense"
                ):
                    for i, example in enumerate(examples):
                        examples[i] = examples[i] * mask
                print("第2步"*3)

                outputs = self.model(examples)

            # Save the ground truth and gradients
            loss = self._loss_criterion(outputs, labels)
            grad = torch.autograd.grad(
                loss,
                self.model.parameters(),
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
            )
            self.list_grad = list((_.detach().clone() for _ in grad))

        self._loss_tracker.update(loss, labels.size(0))

        return loss

    def train_step_end(self, config, batch=None, loss=None):
        """Method called at the end of a training step."""
        # Apply defense if needed
        if hasattr(Config().algorithm, "defense"):
            if Config().algorithm.defense == "GradDefense":
                if (
                    hasattr(Config().algorithm, "clip")
                    and Config().algorithm.clip is True
                ):
                    from defense.GradDefense.clip import noise
                else:
                    from defense.GradDefense.perturb import noise
                self.list_grad = noise(
                    dy_dx=self.list_grad,
                    sensitivity=self.sensitivity,
                    slices_num=Config().algorithm.slices_num,
                    perturb_slices_num=Config().algorithm.perturb_slices_num,
                    noise_intensity=Config().algorithm.scale,
                )

            elif Config().algorithm.defense == "Soteria":
                deviation_f1_target = torch.zeros_like(self.feature_fc1_graph)
                deviation_f1_x_norm = torch.zeros_like(self.feature_fc1_graph)
                for f in range(deviation_f1_x_norm.size(1)):
                    deviation_f1_target[:, f] = 1
                    self.feature_fc1_graph.backward(
                        deviation_f1_target, retain_graph=True,
                    )
                    deviation_f1_x = self.examples.grad.data
                    deviation_f1_x_norm[:, f] = torch.norm(
                        deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1
                    ) / (self.feature_fc1_graph.data[:, f])
                    self.model.zero_grad()
                    self.examples.grad.data.zero_()
                    deviation_f1_target[:, f] = 0

                deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
                thresh = np.percentile(
                    deviation_f1_x_norm_sum.flatten().cpu().numpy(),
                    Config().algorithm.threshold,
                )
                mask = np.where(
                    abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1
                ).astype(np.float32)
                # print(sum(mask))
                self.list_grad[6] = self.list_grad[6] * torch.Tensor(mask).to(
                    self.device
                )

            elif Config().algorithm.defense == "GC" or Config().algorithm.defense == "MaskDefense":
                for i, grad in enumerate(self.list_grad):
                    grad_tensor = grad.cpu().numpy()
                    flattened_weights = np.abs(grad_tensor.flatten())
                    # Generate the pruning threshold according to 'prune by percentage'
                    thresh = np.percentile(
                        flattened_weights, Config().algorithm.prune_pct
                    )
                    grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
                    self.list_grad[i] = torch.Tensor(grad_tensor).to(self.device)

            elif Config().algorithm.defense == "DP":
                for i, grad in enumerate(self.list_grad):
                    grad_tensor = grad.cpu().numpy()
                    noise = np.random.laplace(
                        0, Config().algorithm.epsilon, size=grad_tensor.shape
                    )
                    grad_tensor = grad_tensor + noise
                    self.list_grad[i] = torch.Tensor(grad_tensor).to(self.device)

            elif Config().algorithm.defense == "Outpost":
                iteration = self.current_epoch * (batch + 1)
                # Probability decay
                if random.random() < 1 / (1 + Config().algorithm.beta * iteration):
                    # Risk evaluation
                    risk = compute_risk(self.model)
                    # Perturb
                    from defense.Outpost.perturb import noise

                    self.list_grad = noise(dy_dx=self.list_grad, risk=risk)

            elif Config().algorithm.defense == "FeaDefense":
                print("the length of list_grad:",len(self.list_grad))
                for grad in self.list_grad:
                    print(grad.shape)
                img = self.examples.detach().cpu()
                if Config().data.datasource == "CIFAR10":
                    img = (0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]).unsqueeze(1)
                fea_list = [fea.detach().cpu() for fea in self.fea_graph_list]
                num_layers = len(fea_list)
                cos_value_matrix_3d = []
                for i in range(num_layers):
                    fea_graph = fea_list[i]
                    num_samples = fea_graph.size(0)
                    num_channels = fea_graph.size(1)
                    cos_value_matrix_2d = []
                    for j in range(num_samples):
                        avg_cos = 0
                        cos_value_list = []
                        for k in range(num_channels):
                            o_fea = img[j, 0].numpy()
                            fea = fea_graph[j, k].numpy()
                            fea = cv2.resize(fea, (o_fea.shape[0], o_fea.shape[1]))
                            cos_value = cosine_similarity(o_fea.ravel(), fea.ravel())
                            avg_cos += cos_value
                            cos_value_list.append(cos_value)
                        cos_value_matrix_2d.append(cos_value_list)
                    cos_value_matrix_3d.append(cos_value_matrix_2d)
                    cos_value_matrix_3d.append(cos_value_matrix_2d)
                cos_value_matrix = np.mean(cos_value_matrix_3d, axis=1)
                # print(cos_value_matrix.shape)

                for i, grad in enumerate(self.list_grad):
                    grad_tensor = grad.cpu().numpy()
                    if i >= 6 : continue # 忽略FC层
                    for j in range(grad_tensor.shape[0]):
                        noise = np.random.normal(0, cos_value_matrix[i][j] * Config().algorithm.sigma, size=grad_tensor[j].shape)
                        grad_tensor[j] = grad_tensor[j] + noise
                    self.list_grad[i] = torch.Tensor(grad_tensor).to(self.device)

                # TODO batch img [batch, 3, 32, 32]
                # img = self.examples.detach().cpu()
                # # print("origin img:",img.shape)
                # if Config().data.datasource == "CIFAR10":
                #     img = (0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]).unsqueeze(1)
                # # print("origin img gray:",img.shape)
                # fea_list = [fea.detach().cpu() for fea in self.fea_graph_list]
                # # torch.Size([1, 12, 16, 16])
                # # torch.Size([1, 12, 8, 8])
                # # torch.Size([1, 12, 8, 8])
                #
                # num_layers = len(fea_list)
                # pool_img = img
                # fig, axes = plt.subplots(num_layers, 12, figsize=(4 * 12, 4 * num_layers))
                # cos_value_matrix = []
                # for i in range(num_layers):
                #     fea_graph = fea_list[i]
                #     num_channels = fea_graph.size(1)
                #     # print(f"conv {i+1} channels:",num_channels)
                #     # if i == 0 or i == 1:
                #     #     pool_img = F.avg_pool2d(img, kernel_size=2 * (i + 1), stride=2 * (i + 1))
                #     avg_psnr = 0
                #     avg_ssim = 0
                #     avg_cos = 0
                #     cos_value_list = []
                #     # print("pool_img shape:",pool_img.shape)
                #     for j in range(num_channels):
                #         o_fea = pool_img[0, 0].numpy()
                #         fea = fea_graph[0, j].numpy()
                #         fea = cv2.resize(fea, (o_fea.shape[0], o_fea.shape[1]))
                #
                #         psnr_value = psnr(o_fea, fea, data_range=1.0)
                #         ssim_value = ssim(o_fea, fea, data_range=1.0)
                #         cos_value = cosine_similarity(o_fea.ravel(), fea.ravel())
                #         avg_psnr += psnr_value
                #         avg_ssim += ssim_value
                #         avg_cos += cos_value
                #         cos_value_list.append(cos_value)
                #
                #         axes[i, j].imshow(fea)
                #         axes[i, j].set_title(
                #             f'Conv {i} Channel {j + 1}\n psnr {psnr_value:.4f}\n ssim {ssim_value:.4f}\n cos {cos_value:.4f}',
                #             fontsize=5)
                #         axes[i, j].axis('off')
                #
                #     cos_value_matrix.append(cos_value_list)
                #     cos_value_matrix.append(cos_value_list)
                #     # print("avg_psnr:", avg_psnr)
                #     # print("avg_ssim:", avg_ssim)
                #     # print("avg_cos:", avg_cos)
                # plt.tight_layout()
                # plt.show()
                # # print(len(cos_value_matrix))
                # # print(cos_value_matrix)
                #
                # for i, grad in enumerate(self.list_grad):
                #     grad_tensor = grad.cpu().numpy()
                #     if i >= 6 : continue # 忽略FC层
                #     for j in range(grad_tensor.shape[0]):
                #         noise = np.random.normal(0, cos_value_matrix[i][j] * Config().algorithm.sigma, size=grad_tensor[j].shape)
                #         grad_tensor[j] = grad_tensor[j] + noise
                #     self.list_grad[i] = torch.Tensor(grad_tensor).to(self.device)

            # cast grad back to tuple type
            grad = tuple(self.list_grad)

        # Update model weights with gradients and learning rate
        for param, grad_part in zip(self.model.parameters(), grad):
            param.data = param.data - Config().parameters.optimizer.lr * grad_part

        # Sum up the gradients for each local update
        try:
            self.target_grad = [
                sum(x)
                for x in zip(list((_.detach().clone() for _ in grad)), self.target_grad)
            ]
        except:
            self.target_grad = list((_.detach().clone() for _ in grad))

    def train_run_end(self, config, **kwargs):
        """Method called at the end of a training run."""
        if (
            hasattr(Config().algorithm, "share_gradients")
            and Config().algorithm.share_gradients
        ):
            try:
                total_local_steps = config["epochs"] * math.ceil(
                    Config().data.partition_size / config["batch_size"]
                )
                self.target_grad = [x / total_local_steps for x in self.target_grad]
            except:
                self.target_grad = None

        self.full_examples = self.full_examples.detach()
        file_path = f"{Config().params['model_path']}/{self.client_id}.pickle"
        with open(file_path, "wb") as handle:
            pickle.dump(
                [self.full_examples, self.full_onehot_labels, self.target_grad], handle
            )

    @staticmethod
    def process_outputs(outputs):
        """
        Method called after the model updates have been generated.
        """
        return outputs[0]
