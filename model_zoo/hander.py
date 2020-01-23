import os
import datetime
import time
from sklearn import metrics
from setting import logger
import torch
import pickle as pkl
import torch.nn.functional as F
import torchsnooper


class BaseHander(object):
    def __init__(self, args):
        pass
    
    def evl_index(self, label, pred, detail=False):
        """
        metrics：
        f1, accuracy, precision, recall
        confusion_matrix
        """
        f1 = metrics.f1_score(label, pred)
        acc = metrics.accuracy_score(label, pred)
        precision = metrics.precision_score(label, pred)
        recall = metrics.recall_score(label, pred)
        logger.info("F1={:.4f}\tACC={:.4f}\tPrecision={:.4f}\tRecall={:.4f}.".format(f1, acc, precision, recall))
        if detail is True:
            logger.info("\nConfusion_Matrix:\n{}".format(metrics.confusion_matrix(label, pred)))
        return f1, acc, precision, recall

    def save_mode(self, model_path, report_file):
        f1, acc, precision, recall = self.hist_evl_index[-1]

        checkpoint = dict()
        checkpoint['state_dict'] = self.model.state_dict()
        checkpoint['hist_loss'] = self.hist_loss
        checkpoint['hist_evl_index'] = self.hist_evl_index 
        model_name = self.model.__class__.__name__
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H")
        if self.view is None:
            file = "{}_{}_f1{:.4f}".format(timestamp, 'app_han_app', f1)
        else:
            file = "{}_{}_f1{:.4f}".format(timestamp, self.view, f1)
        torch.save(checkpoint, os.path.join(model_path, file))
        logger.info("save model and train history to {}".format(os.path.join(model_path, file)))

        with open(report_file, 'a') as f:
            f.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\n".format(model_name, f1, acc, precision, recall, file))
    
    def get_embedding(self, embedding_path):
        node, label = self.inputdata.get_test_data()
        embed = self.model.get_embedding(node)
        ret = dict()
        ret['embedding'] = embed.data.numpy()
        ret['label'] = label
        model_name = self.model.__class__.__name__
        if not os.path.exists(os.path.join(embedding_path, model_name)):
            os.mkdir(os.path.join(embedding_path, model_name))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H")
        with open(os.path.join(embedding_path, model_name, timestamp), 'wb') as f:
            pkl.dump(ret, f)
        logger.info("save embeddings of test dataset to {}".format(os.path.join(embedding_path, model_name, timestamp)))
    
    
    def train(self, epoch=10, interval_val=10):
        self.build_model()
        self.hist_loss = []
        self.hist_evl_index = []
        iter = 0
        for ep in range(epoch):
            for x, y in self.train_data_loader:
                iter += 1
                self.model.train()
                nodes = [i.item() for i in x.squeeze_()]
                labels = y.squeeze_()
                start_time = time.time()
                out = self.model.forward(nodes)
                loss = self.loss_func(out, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.hist_loss.append(loss.item())
                logger.info("epoch{}[iter{}]:{:.5f}s\tloss={:.5f}".format(ep, iter, time.time()-start_time, loss.item()))
                
                if iter % interval_val == 0:
                    self.model.eval()
                    logger.info("Eveluation on val dataset:")
                    val_x, val_y = self.inputdata.get_val_data()
                    val_output = self.model.forward(val_x)
                    self.hist_evl_index.append(self.evl_index(val_y, val_output.cpu().data.numpy().argmax(axis=1), detail=False))

        logger.info("optimization finished! Eveluation on test dataset:")
        self.model.eval()
        test_x, test_y = self.inputdata.get_test_data()
        test_output = self.model.forward(test_x)

        pred_prob = F.softmax(test_output, dim=1).cpu().data.numpy()[:,1]
        fpr, tpr, thresthods = metrics.roc_curve(test_y, pred_prob)
        auc = metrics.auc(fpr, tpr)
        print("AUC={:.4f}".format(auc))

        self.hist_evl_index.append(self.evl_index(test_y, test_output.cpu().data.numpy().argmax(axis=1), detail=True))
        
        return self.hist_evl_index[-1]
        
    def load_checkpoint(self, checkpoint_file):
        self.build_model()
        checkpoint = torch.load(checkpoint_file)
        logger.info("==================")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        logger.info("load pretrain model from {}".format(checkpoint_file))
