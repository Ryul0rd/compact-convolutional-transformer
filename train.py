from pytorch_lightning import Trainer
from pl_bolts.datamodules import MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule
from compact_conv_transformer import CCTLitMod

def main():
    trainer = Trainer(max_epochs=200, gpus=1)
    model = CCTLitMod(n_channels=3)
    data = CIFAR10DataModule(num_workers=4)
    trainer.fit(model, data)

if __name__ == '__main__':
    main()