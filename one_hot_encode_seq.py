import numpy as np

max_len = 12 # maximum peptide length
k = 5 # k-mer length

pad = '_'
alphabet = ['V', 'H', 'L', 'A', 'N', 'P', 'Y', 'W', 'T', 'E', 'Q', 'G', 'M', 'S', 'R', 'D', 'C', 'I', 'K', 'F',pad]

### MODIFICATIONS #############################################
mod_aa_alphabet = ['O']
delta_masses = [15.99491463]
assert(len(set(alphabet).intersection(set(mod_aa_alphabet))) == 0)
massdelta_to_char = dict(zip(delta_masses,mod_aa_alphabet))
aa_alphabet = alphabet + mod_aa_alphabet
###############################################################
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

def padding(seq,max_len=12,pad=pad):
  n = len(seq)
  if n < max_len:
    return seq+pad*(max_len-n)
  else:
    return seq[:max_len]

def kmers(seq, k):
  n = len(seq)
  #result = [] 
  #for i in range(n-k+1):
  #  result.append(seq[i:i+k])
  result = list(map(lambda i: seq[i:i+k],range(n-k+1)))
  return result

def integrate_modifications(seq,mod_list,massdelta_to_char=massdelta_to_char):
  '''
  Inputs: 
  seq -  pepitde sequence e.g. 'TKMPYMGMA'
  mod_lost -  list of tuples: (pos,mod_type)
    [(pos1, mass1), (pos2, mass2)] 
    [(3, 15.99491463), (8, 15.99491463)]

  Output:
  seq : peptide sequence with exchanged characters e.g. 'M'+delta_mass -> 'O'
  e.g. 'TKMPYMGMA' becomes 'TKOPYMGOA' 
  '''       
  for loc, mass in mod_list:
      seq = list(seq)
      seq[loc-1] = massdelta_to_char[mass]
      seq = ''.join(seq)
  return seq

def do_all_the_shit(seq,max_len=max_len,k=k):
  '''
  Do: padding+kmer+indices+onehot
  Input:
  sequence
  Output:
  array with shape: (n-k+1,k,alphabet_size)
  '''
  padded = padding(seq,max_len=max_len) # add padding
  kmer = kmers(padded,k) # create list of kmers
  kmer_indices= list(map(lambda char: to_index(char),kmer)) # transform list of kmers to indices
  kmer_one_hot = list(map(lambda index: one_hot(index),kmer_indices)) # one-hot encode
  return np.array(kmer_one_hot)

if __name__ == "__main__":
        
    seq = 'CGICGCGDTGEHHHEHEHEH' 
    sequences = ['CGICGCGDTGEHHHEHEHEH' ,'TKMPYMGMA']

    for seq in sequences:#map(lambda x: x,sequences): 
      max_len = 30
      k = 5            
      padded = padding(seq,max_len=max_len)
      kmer = kmers(padded,k)
      kmer_indices= list(map(lambda char: to_index(char),kmer))
      kmer_one_hot = list(map(lambda index: one_hot(index),kmer_indices))
      print(np.array(kmer_one_hot).shape)
      print(kmer_indices)

    ### Try encoding
    indices = to_index(seq,aa_to_index)
    one_hot_encoded = one_hot(indices,len(aa_alphabet))
    
    reversed_indices = rev_one_hot(one_hot_encoded)
    reversed_seq = to_aa(reversed_indices,index_to_aa)
    print(reversed_indices)
    print(indices)
    print(seq)
    print(''.join(reversed_seq))

    ##### Modifications: 
    print(integrate_modifications('TKMPYMGMA',[(3, 15.99491463), (8, 15.99491463)]))