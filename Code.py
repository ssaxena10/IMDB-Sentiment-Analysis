
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request

def download_data():
    
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()

def read_data(path):
   
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])

def tokenize(doc, keep_internal_punct=False):

    Ldoc = doc.lower()
    if keep_internal_punct == True:
        return np.array([d.strip(string.punctuation) for d in Ldoc.split()])

    elif keep_internal_punct == False:
        return np.array(re.sub('\W+',' ', Ldoc).split())

    pass

def token_features(tokens, feats):

    c = Counter(tokens)
    for i in c:
        k = ''.join(["token=",i])
        feats[k] = c[i]
    pass


def token_pair_features(tokens, feats, k=3):

    tpf = []
    r = len(tokens)-k+1
    for x in range(r):
        for y in combinations(tokens[x:x+k],2):

            tpf.extend(["=".join(["token_pair","__".join(y)])])

    for x in tpf:
        feats[x] += 1

    pass

neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):

    p=0
    n=0
    for i in tokens:
        i = i.lower()
        if i in neg_words:
            n += 1
        if i in pos_words:
            p += 1
    feats['neg_words'] = n
    feats['pos_words'] = p

    pass

def featurize(tokens, feature_fns):

    feats = defaultdict(int)
    for tup in feature_fns:
        tup(tokens,feats)
    return sorted(feats.items())

    pass

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):

    feat=[]
    row = 0
    tcount = defaultdict(int)
    matdata,matrow,matcol = [],[],[]

    for t in tokens_list:
        f = featurize(t,feature_fns)
        feat.append(f)
        for x in f:
            if(x not in tcount):
                tcount[x[0]]+=1

    if(vocab == None):
        vocab = defaultdict(int)
        i=0
        for x in sorted(tcount.items()):
            if(x[1]>=min_freq):
                vocab[x[0]]=i
                i=i+1

    for t in tokens_list:
        for x in sorted(feat[row]):
            if(x[0] in vocab.keys()):
                matrow.append(row)
                matcol.append(vocab[x[0]])
                matdata.append(x[1])
        row+=1

    Mat = csr_matrix((matdata,(matrow,matcol)),dtype=np.int64)
    return Mat,vocab
    pass


def accuracy_score(truth, predicted):

    return len(np.where(truth==predicted)[0]) / len(truth)

def cross_validation_accuracy(clf, X, labels, k):

    cva = KFold(len(labels), k)
    alist = []
    for train, test in cva:
        clf.fit(X[train], labels[train])
        alist.append(accuracy_score(labels[test], clf.predict(X[test])))

    return np.mean(alist)
    pass


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):

    flist = []
    feats = []
    for x in range(1,len(feature_fns)+1):
        for i in combinations(feature_fns,x):
            feats.append(i)
    for pv in punct_vals:
        tlist = []
        for d in docs:
            tlist.append(tokenize(d,pv))
        for m in min_freqs:
            for f in feats:
                Mat, vocab = vectorize(tlist, f, m)
                a=cross_validation_accuracy(LogisticRegression(), Mat, labels, 5)
                flist.append({'punct' : pv, 'features' : f, 'min_freq' : m, 'accuracy' : a})
    return sorted(flist,key=lambda x:(x['accuracy'],x['min_freq']), reverse=True)

    pass

def plot_sorted_accuracies(results):

    alist=[]
    for i in results:
        alist.append(i['accuracy'])
    alist= sorted(alist)
    plt.plot(alist)
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.savefig("accuracies.png")
    pass

def mean_accuracy_per_setting(results):

    maps=[]
    for r in results:
        maps.append(("min_freq",r['min_freq']))
        maps.append(("features",r['features']))
        maps.append(("punct",r['punct']))
    result = []
    for m in set(maps):
        acc = []
        for r in results:
            if(r[m[0]]==m[1]):
                acc.append(r['accuracy'])
        if(m[0]=='features'):
            f=[]
            f.append(m[0]+"=")
            for i in m[1]:
                f.append(i.__name__)
            st = " ".join(f)
            result.append((np.mean(acc),st))
        else:
            result.append((np.mean(acc),(m[0] + "=" + str(m[1]))))
    return sorted(result,key=lambda x:-x[0])
    pass

def fit_best_classifier(docs, labels, best_result):

    tlist = []
    for d in docs:
        tlist.append(tokenize(d,best_result['punct']))

    Mat,vocab=vectorize(tlist,list(best_result['features']), best_result['min_freq'])
    clf = LogisticRegression()
    clf.fit(Mat, labels)
    return clf,vocab
    pass

def top_coefs(clf, label, n, vocab):

    pos=[]
    neg=[]
    c = clf.coef_[0]
    for key,val in zip(sorted(vocab.keys()), c):

        if val>=0:
            pos.append((key,val))
        else:
            neg.append((key,abs(val)))

    if label == 1:
        return sorted(pos, key=lambda x:-x[1])[:n]
    else:
        return sorted(neg, key=lambda x:-x[1])[:n]
    pass

def parse_test_data(best_result, vocab):

    testD, testL = read_data(os.path.join('data', 'test'))
    dval=[tokenize(x,best_result['punct']) for x in testD]

    Mat,vocab=vectorize(dval,list(best_result['features']), best_result['min_freq'],vocab)

    return testD,testL,Mat
    pass

def print_top_misclassified(test_docs, test_labels, X_test, clf, n):

    prb = clf.predict_proba(X_test)
    m = defaultdict(int)
    prd = clf.predict(X_test)
    for i in range(len(prd)):
        if prd[i] != test_labels[i]:
            m[i] = prb[i][prd[i]]

    m=sorted(m.items(), key=lambda x: x[1], reverse = True)[:n]
    for x in m:
        print("\ntruth=", test_labels[x[0]], " predicted=", prd[x[0]], " proba=", prb[x[0], prd[x[0]]])
        print(test_docs[x[0]])

    pass

def main():
    
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
