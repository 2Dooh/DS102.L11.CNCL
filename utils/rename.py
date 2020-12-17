import os
import re
import click

@click.command()
@click.option('--extension', '-ext', required=True)
@click.option('--path', '-p', required=True)
def cli(extension, path):

    path = path

    images = os.listdir(path)

    pattern = re.compile('(\d+).{}'.format(extension))

    indices = list(map(lambda img: int(re.findall(pattern, img)[0]), images))

    for idx, img in zip(indices, images):
        filename = os.path.join(path, img)
        replaced_name = os.path.join(path, '{:0>3d}.{}'.format(idx, extension))
        os.replace(filename, replaced_name)

if __name__ == "__main__":
    pass
    # cli()