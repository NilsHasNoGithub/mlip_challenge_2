import pickle
import torch
import click
from library.models.timm_model import TimmModule


@click.command()
@click.option("--in-file", "-i", type=click.Path(exists=True), help="input checkpoint file")
@click.option("--out-file", "-o", type=click.Path(), help="output pt file")
def main(
    in_file,
    out_file,
):
    model = TimmModule.load_from_checkpoint(in_file)
    torch.save(model, out_file)
    # with open(out_file, "wb") as f:
        # pickle.dump(model, f)


if __name__ == "__main__":
    main()