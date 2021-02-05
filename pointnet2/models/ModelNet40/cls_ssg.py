from pointnet2.models.common.pointnet2_ssg_cls import PointNet2ClassificationSSG

class ModelNet40PointNet2ClassificationSSG(PointNet2ClassificationSSG):
    def __init__(self, hparams):
        super().__init__(hparams)
    
    def validation_step(self, batch, batch_idx):
        pc, labels = batch
        logits = self.forward(pc)
        loss = F.cross_entropy(logits, labels)
        acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        return dict(val_loss=loss, val_acc=acc)