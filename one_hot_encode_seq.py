import numpy as np

aa_alphabet = ['V', 'H', 'L', 'A', 'N', 'P', 'Y', 'W', 'T', 'E', 'Q', 'G', 'M', 'S', 'R', 'D', 'C', 'I', 'K', 'F']
aa_index = range(len(aa_alphabet))
aa_to_index = dict(zip(aa_alphabet,aa_index))
index_to_aa = dict(zip(aa_index,aa_alphabet))

def to_index(seq,aa_to_index=aa_to_index):
    return np.array(list(map(lambda c: aa_to_index[c], list(seq))))

def one_hot(a, num_classes=len(aa_alphabet)):
    a = np.array(a)  
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def rev_one_hot(one_hot_vec):
    return np.argmax(one_hot_encoded, axis=-1)

def to_aa(indices,index_to_aa=index_to_aa):
    return np.array(list(map(lambda c: index_to_aa[c], list(indices))))

if __name__ == "__main__":
        
    seq = 'CGICGCGDTGEHHHEHEHEH'

    indices = to_index(seq,aa_to_index)
    one_hot_encoded = one_hot(indices,len(aa_alphabet))
    
    reversed_indices = rev_one_hot(one_hot_encoded)
    reversed_seq = to_aa(reversed_indices,index_to_aa)
    print(reversed_indices,indices)
    print(seq,''.join(reversed_seq))