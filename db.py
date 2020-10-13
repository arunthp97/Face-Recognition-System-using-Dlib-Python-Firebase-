import numpy as np
import cloudpickle as cPickle

def add_embeddings(embedding, label,label1,label2,label3,label4,
                   embeddings_path="face_embeddings.npy",
                   labels_path="labels.cpickle",
                   labels_path1="labels1.cpickle",
                   labels_path2="labels2.cpickle",
                   labels_path3="labels3.cpickle",
                   labels_path4="labels4.cpickle"):
    first_time = False
    try:
        embeddings = np.load(embeddings_path)
        labels = cPickle.load(open(labels_path,"rb"))
        labels1 = cPickle.load(open(labels_path1,"rb"))
        labels2 = cPickle.load(open(labels_path2,"rb"))
        labels3 = cPickle.load(open(labels_path3,"rb"))
        labels4 = cPickle.load(open(labels_path4,"rb"))
    except IOError:
        first_time = True

    if first_time:
        embeddings = embedding
        labels = [label]
        labels1 = [label1]
        labels2 = [label2]
        labels3 = [label3]
        labels4 = [label4]
    else:
        embeddings = np.concatenate([embeddings, embedding], axis=0)
        labels.append(label)
        labels1.append(label1)
        labels2.append(label2)
        labels3.append(label3)
        labels4.append(label4)

    np.save(embeddings_path, embeddings)
    with open(labels_path, "wb") as f:
        cPickle.dump(labels, f)
    with open(labels_path1, "wb") as f:
        cPickle.dump(labels1, f)
    with open(labels_path2, "wb") as f:
        cPickle.dump(labels2, f)
    with open(labels_path3, "wb") as f:
        cPickle.dump(labels3, f)
    with open(labels_path4, "wb") as f:
        cPickle.dump(labels4, f)

    return True