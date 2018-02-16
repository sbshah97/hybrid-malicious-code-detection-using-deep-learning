
# coding: utf-8

# In[2]:


from pandas import read_csv, DataFrame
from numpy.random import seed


# In[3]:


from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense
from keras.models import Model

# In[4]:


df = read_csv("kddcup.data", index_col=None)


# # Dataset mapping for string columns

# * protocol_type 
# ```python
# pd['protocol_type'].unique()
# array(['tcp', 'udp', 'icmp'], dtype=object)
# ```
# 
# * service
# ```python
# >>> pd['service'].unique()
# array(['http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp',
#        'eco_i', 'ntp_u', 'ecr_i', 'other', 'private', 'pop_3', 'ftp_data',
#        'rje', 'time', 'mtp', 'link', 'remote_job', 'gopher', 'ssh',
#        'name', 'whois', 'domain', 'login', 'imap4', 'daytime', 'ctf',
#        'nntp', 'shell', 'IRC', 'nnsp', 'http_443', 'exec', 'printer',
#        'efs', 'courier', 'uucp', 'klogin', 'kshell', 'echo', 'discard',
#        'systat', 'supdup', 'iso_tsap', 'hostnames', 'csnet_ns', 'pop_2',
#        'sunrpc', 'uucp_path', 'netbios_ns', 'netbios_ssn', 'netbios_dgm',
#        'sql_net', 'vmnet', 'bgp', 'Z39_50', 'ldap', 'netstat', 'urh_i',
#        'X11', 'urp_i', 'pm_dump', 'tftp_u', 'tim_i', 'red_i'],
#       dtype=object)
# ```
# 
# * flag
# ```python
# >>> pd['flag'].unique()
# array(['SF', 'S1', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0',
#        'OTH', 'SH'], dtype=object)
# ```
# * result
# ```python
# >>> pd['result'].unique()
# array(['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.',
#        'smurf.', 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.',
#        'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.',
#        'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
#        'spy.', 'rootkit.'], dtype=object)
# ```

# In[5]:


protocol_mapping = {'tcp':1, 'udp':2, 'icmp':3}


# In[6]:


service_mapping = {'http':1, 'smtp':2, 'finger':3, 'domain_u':4, 'auth':5, 'telnet':6, 'ftp':7,
       'eco_i':8, 'ntp_u':9, 'ecr_i':10, 'other':11, 'private':12, 'pop_3':13, 'ftp_data':14,
       'rje':15, 'time':16, 'mtp':17, 'link':18, 'remote_job':19, 'gopher':20, 'ssh':21,
       'name':22, 'whois':23, 'domain':24, 'login':25, 'imap4':26, 'daytime':27, 'ctf':28,
       'nntp':29, 'shell':30, 'IRC':31, 'nnsp':32, 'http_443':33, 'exec':34, 'printer':35,
       'efs':36, 'courier':37, 'uucp':38, 'klogin':39, 'kshell':40, 'echo':41, 'discard':42,
       'systat':43, 'supdup':44, 'iso_tsap':45, 'hostnames':46, 'csnet_ns':47, 'pop_2':48,
       'sunrpc':49, 'uucp_path':50, 'netbios_ns':51, 'netbios_ssn':52, 'netbios_dgm':53,
       'sql_net':54, 'vmnet':55, 'bgp':56, 'Z39_50':57, 'ldap':58, 'netstat':59, 'urh_i':60,
       'X11':61, 'urp_i':62, 'pm_dump':63, 'tftp_u':64, 'tim_i':65, 'red_i':66}


# In[7]:


flag_mapping = {'SF':1, 'S1':2, 'REJ':3, 'S2':4, 'S0':5, 'S3':6, 'RSTO':7, 'RSTR':8, 'RSTOS0':9, 'OTH':10, 'SH':11}


# In[8]:


result_mapping = {'normal.':1, 'buffer_overflow.':2, 'loadmodule.':3, 'perl.':4, 'neptune.':5, 'smurf.':6, 'guess_passwd.':7, 'pod.':8, 'teardrop.':9, 'portsweep.':10,
     'ipsweep.':11, 'land.':12, 'ftp_write.':13, 'back.':14, 'imap.':15, 'satan.':16,
     'phf.':17, 'nmap.':18, 'multihop.':19, 'warezmaster.':20, 'warezclient.':21,
     'spy.':22, 'rootkit.':23}


# In[9]:


df1 = df.replace({'protocol_type': {'tcp':1, 'udp':2, 'icmp':3}})
df2 = df1.replace({'flag': {'SF':1, 'S1':2, 'REJ':3, 'S2':4, 'S0':5, 'S3':6, 'RSTO':7, 'RSTR':8, 'RSTOS0':9, 'OTH':10, 'SH':11}})

df3 = df2.replace({'service': {'http':1, 'smtp':2, 'finger':3, 'domain_u':4, 'auth':5, 'telnet':6, 'ftp':7,
       'eco_i':8, 'ntp_u':9, 'ecr_i':10, 'other':11, 'private':12, 'pop_3':13, 'ftp_data':14,
       'rje':15, 'time':16, 'mtp':17, 'link':18, 'remote_job':19, 'gopher':20, 'ssh':21,
       'name':22, 'whois':23, 'domain':24, 'login':25, 'imap4':26, 'daytime':27, 'ctf':28,
       'nntp':29, 'shell':30, 'IRC':31, 'nnsp':32, 'http_443':33, 'exec':34, 'printer':35,
       'efs':36, 'courier':37, 'uucp':38, 'klogin':39, 'kshell':40, 'echo':41, 'discard':42,
       'systat':43, 'supdup':44, 'iso_tsap':45, 'hostnames':46, 'csnet_ns':47, 'pop_2':48,
       'sunrpc':49, 'uucp_path':50, 'netbios_ns':51, 'netbios_ssn':52, 'netbios_dgm':53,
       'sql_net':54, 'vmnet':55, 'bgp':56, 'Z39_50':57, 'ldap':58, 'netstat':59, 'urh_i':60,
       'X11':61, 'urp_i':62, 'pm_dump':63, 'tftp_u':64, 'tim_i':65, 'red_i':66}})

df = df3.replace({'result': {'normal.':1, 'buffer_overflow.':2, 'loadmodule.':3, 'perl.':4, 'neptune.':5, 'smurf.':6, 'guess_passwd.':7, 'pod.':8, 'teardrop.':9, 'portsweep.':10,
     'ipsweep.':11, 'land.':12, 'ftp_write.':13, 'back.':14, 'imap.':15, 'satan.':16,
     'phf.':17, 'nmap.':18, 'multihop.':19, 'warezmaster.':20, 'warezclient.':21,
     'spy.':22, 'rootkit.':23}})
df.head()


# In[13]:


X = df.iloc[:, 1:-1]
Y = df.result
X.head()


# In[14]:


# Scale value of x
sX = minmax_scale(X, axis = 0)
ncol = sX.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(sX, Y, train_size = 0.7, random_state = seed(2017))


# In[15]:


# ## Simple Autoencoder
# input_dim = Input(shape = (ncol, ))
# # DEFINE THE DIMENSION OF ENCODER ASSUMED 3
# encoding_dim = 20
# # DEFINE THE ENCODER LAYER
# encoded = Dense(encoding_dim, activation = 'relu')(input_dim)
# # DEFINE THE DECODER LAYER
# decoded = Dense(ncol, activation = 'sigmoid')(encoded)
# # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
# autoencoder = Model(input = input_dim, output = decoded)
# # CONFIGURE AND TRAIN THE AUTOENCODER
# autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
# autoencoder.fit(X_train, X_train, nb_epoch = 50, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))

input_dim = Input(shape = (ncol, ))
# DEFINE THE DIMENSION OF ENCODER ASSUMED 3
encoding_dim = 20
# DEFINE THE ENCODER LAYERS
encoded1 = Dense(300, activation = 'relu')(input_dim)
encoded2 = Dense(150, activation = 'relu')(encoded1)
encoded3 = Dense(75, activation = 'relu')(encoded2)
encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)
# DEFINE THE DECODER LAYERS
decoded1 = Dense(75, activation = 'relu')(encoded4)
decoded2 = Dense(150, activation = 'relu')(decoded1)
decoded3 = Dense(300, activation = 'relu')(decoded2)
decoded4 = Dense(ncol, activation = 'sigmoid')(decoded3)
# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(input = input_dim, output = decoded4)
# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs = 100, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
# THE 