
from agents import *
import json
import time
import click
import os
import datetime

@click.command()
@click.option('--config', required=True)
@click.option('--seed', default=0)
def cli(config, seed):
    with open('./configs/{}'.format(config)) as  json_file:
        config = json.load(json_file)
    agent_constructor = globals()[config['agent']]

    agent = agent_constructor(**config, seed=seed)
    start = time.time()

    agent.run()
    agent.finalize()
    end = time.time() - start

    print('Elapsed time: {}'.format(end))

if __name__ == '__main__':
    cli()