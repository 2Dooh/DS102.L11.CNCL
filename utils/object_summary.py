import json
import xmltodict
import os
import click
import numpy as np

@click.command()
@click.option('--annot_path', required=True)
def cli(annot_path):
    widths, heights, labels = [], [], []
    for annot in os.listdir(annot_path):
        with open(os.path.join(annot_path, annot)) as handle:
            ordered_dict = xmltodict.parse(handle.read())
        annotation = json.loads(json.dumps(ordered_dict)) 
        objects = annotation['annotation']['object']
        for obj in objects:
            label, coords = obj['name'], obj['bndbox']
            [xmin, ymin, xmax, ymax] = [int(coord) for coord in coords.values()]

            labels += [label]
            widths += [xmax - xmin]
            heights += [ymax - ymin]

    labels, widths, heights = [np.array(lst) for lst in [labels, widths, heights]]

    stats = '{}: max: {} - min: {} - mean: {} - std: {}'
    print(stats.format('W', widths.max(), widths.min(), widths.mean(), widths.std()))
    print(stats.format('H', heights.max(), heights.min(), heights.mean(), heights.std()))
    print(dict(zip(*[res.tolist() for res in np.unique(labels, return_counts=True)])))
    # print([res.tolist() for res in np.unique(labels, return_counts=True)])
            
if __name__ == '__main__':
    cli()