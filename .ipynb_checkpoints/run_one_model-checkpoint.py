from kan import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


def run_one_model(random_state, x, y, device, prune_remain_num=30):
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)
    dataset = {}
    dtype = torch.get_default_dtype()
    dataset['train_input'] = torch.from_numpy(x_train).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(x_test).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(y_train).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(y_test).type(torch.long).to(device)

    # Define the model, here grid_range is set to [-10, 10] since the input data's range from histgram.
    model = KAN(width=[2000, 5, 4], grid=3, k=3, device=device, grid_range=[-10, 10])

    # Define the metrics
    def train_acc():
        return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype))

    def test_acc():
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype))

    def train_auc():
        model.eval()
        with torch.no_grad():
            # 获取训练集的预测概率
            y_pred = F.softmax(model(dataset['train_input']), dim=1).cpu().numpy()
            y_true = dataset['train_label'].cpu().numpy()
            # 计算每个类别的 ROC AUC 分数，并求平均
            roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
        model.train()
        return roc_auc

    def test_auc():
        model.eval()
        with torch.no_grad():
            # 获取测试集的预测概率
            y_pred = F.softmax(model(dataset['test_input']), dim=1).cpu().numpy()
            y_true = dataset['test_label'].cpu().numpy()
            # 计算每个类别的 ROC AUC 分数，并求平均
            roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
        model.train()
        return roc_auc

    results = model.fit(dataset, opt="LBFGS", steps=10, metrics=(train_acc, test_acc, train_auc, test_auc),
                        loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-2, lamb=0.002, lamb_entropy=10, lamb_coef=1)

    # Get the metrics
    train_accuracy, test_accuracy = train_acc(), test_acc()
    train_roc_auc, test_roc_auc = train_auc(), test_auc()

    # Prune the model
    model.attribute()
    input_score = model.node_scores[0]
    top_n_idx = torch.topk(input_score, prune_remain_num).indices
    threshold = input_score[top_n_idx[-1]]
    top_n_idx = top_n_idx.cpu().numpy()

    model = model.prune()
    model = model.prune_input(threshold=threshold)

    # Retrain the model
    model.fit(dataset, opt="Adam", steps=30, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss(),
              lr=1e-3, lamb=0.001, update_grid=False, singularity_avoiding=True)

    # Get the metrics after pruning
    after_prune_train_accuracy, after_prune_test_accuracy = train_acc(), test_acc()
    after_prune_train_roc_auc, after_prune_test_roc_auc = train_auc(), test_auc()

    return (
        input_score,
        (train_accuracy, train_roc_auc),
        (test_accuracy, test_roc_auc),
        (after_prune_train_accuracy, after_prune_train_roc_auc),
        (after_prune_test_accuracy, after_prune_test_roc_auc)
    )
