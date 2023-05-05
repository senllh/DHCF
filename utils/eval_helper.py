from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score


"""
	Utility functions for evaluating the model performance
"""


def eval_deep(log, loader):
	"""
	Evaluating the classification performance given mini-batch data
	"""

	# get the empirical batch_size for each mini-batch

	data_size = len(loader.dataset.indices)
	batch_size = loader.batch_size
	# print(data_size, batch_size)
	if data_size % batch_size == 0:
		size_list = [batch_size] * (data_size//batch_size)
	else:
		size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size]
	# size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size]
	# print(log, size_list)
	# print(len(log), len(size_list))
	assert len(log) == len(size_list)

	accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

	prob_log, label_log = [], []

	for batch, size in zip(log, size_list):
		pred_y, y = batch[0].data.cpu().numpy().argmax(axis=1), batch[1].data.cpu().numpy().tolist()
		prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
		label_log.extend(y)

		accuracy += accuracy_score(y, pred_y) * size
		f1_macro += f1_score(y, pred_y, average='macro') * size
		f1_micro += f1_score(y, pred_y, average='micro') * size
		precision += precision_score(y, pred_y, zero_division=0) * size
		recall += recall_score(y, pred_y, zero_division=0) * size

	auc = roc_auc_score(label_log, prob_log)
	ap = average_precision_score(label_log, prob_log)

	return accuracy/data_size, f1_macro/data_size, f1_micro/data_size, precision/data_size, recall/data_size, auc, ap

def acc_f1(output, labels, average='binary'):
	preds = output.max(1)[1].type_as(labels)
	if preds.is_cuda:
		preds = preds.cpu()
		labels = labels.cpu()
	accuracy = accuracy_score(preds, labels)
	f1 = f1_score(preds, labels, average=average)
	return accuracy, f1

def metrics(output, labels, average='binary'):
	preds = output.max(1)[1].type_as(labels)
	if preds.is_cuda:
		preds = preds.cpu()
		labels = labels.cpu()
	accuracy = accuracy_score(preds, labels)
	f1 = f1_score(preds, labels, average=average)
	f1_micro = f1_score(preds, labels, average='micro')
	precision = precision_score(preds, labels)
	recall = recall_score(preds, labels)
	# return accuracy, f1
	return accuracy, f1, f1_micro, precision, recall