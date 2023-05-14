from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get_callbacks(
        ckpt_dirpath: str,
        name: str
):
    ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val/avg_c_y_acc',
        mode='max',
        dirpath=ckpt_dirpath,
        filename=f'{name}_best',
        save_last=True
    )

    estp_callback = EarlyStopping(
        monitor='val/t+c_loss',
        min_delta=0.0,
        patience=15,
        verbose=False,
        mode="min",
    )

    return [
        ckpt_callback,
        estp_callback
    ]
