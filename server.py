"""
Running the Plato federated learning emulator.
"""

import asyncio
import logging
import argparse
import websockets

import config
import servers

# Setting up the parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.conf',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s',
    level='INFO', datefmt='%H:%M:%S')


def main():
    """Run a federated learning server."""
    # Read runtime parameters from a configuration file
    fl_config = config.Config(args.config)

    # Initialize the federated learning server
    server = {
        "fedavg": servers.fedavg.FedAvgServer
    }[fl_config.training.server](fl_config)

    server.configure()

    logging.info("Starting the federated learning server...")
    start_server = websockets.serve(server.serve,
                     fl_config.server.address, fl_config.server.port)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()

if __name__ == "__main__":
    main()