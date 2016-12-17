import scipy.io as spio
import scipy.optimize as op
import numpy as np
import pdb
import itertools
import pickle
import sys
import maxflow
RELEASE = 'release'
DEBUG = 'debug'
if len(sys.argv) < 3:
    print "Usage: python crf_solver.py [release/debug] [train/test]"
    sys.exit()

mode =  sys.argv[1]
oper = sys.argv[2]


ITER_ROUND = 10
class CRF_SOLVER(object):
    def __init__(self, config):
        self.grid = np.array(config['grid'])
        self.histo = np.array(config['histo'])
        self.edges = np.array(config['edges'])
        self.mapper = np.array(config['mapper'])
        self.states = 2
        (self.n_nodes, _) = np.shape(self.mapper)
        self._lambda = 0.5
        self.rss_start = -30
        self.rss_end = 30
        self.rss_step = 0.5
        self.nbins = (self.rss_end - self.rss_start)/self.rss_step

    def rss_range(self, val):
        return int(np.floor(( val - self.rss_start) / (self.rss_end - self.rss_start ) * self.nbins )) 
        
    def objective_and_gradients_batch(self, w, seqs, labels, flag):
        (sample_len, dim) = np.shape( seqs )
        logl = 0.
        sigma2 = 10
        lambda_ = 1./2/sigma2
        g = np.zeros(len(w))
        for i in xrange(sample_len):
            cur_seq = seqs[i]
            cur_label = labels[i]
            if mode == DEBUG:
                pdb.set_trace()
            ## compute M
            (_, label_len) = np.shape(labels)
            Md = self.compute_M(self.grid, w, self.edges, cur_seq, label_len, self.states)
            if mode == DEBUG:
                pdb.set_trace()
            ## compute alpha score
            alpha = self.compute_alpha_score(self.grid, self.states, Md)
            if mode == DEBUG:
                pdb.set_trace()
            ## compute beta score
            beta = self.compute_beta_score(self.grid, self.states, Md)
            ## compute normalize term
            if mode == DEBUG:
                pdb.set_trace()
            Z = 1.
            for M in Md:
               Z = np.dot(Z,M)
            if mode == DEBUG:
                pdb.set_trace()
            Z = Z[0][0]
            if Z == 0:
                pdb.set_trace()
            ## compute marginal probability
            prod, pairwise_prod = self.compute_marginal(self.grid, self.edges, self.states, Z, alpha, beta, Md) 
            if mode == DEBUG:
                pdb.set_trace()
            ## compute 
            ## compute gradient
            g_tmp,logp = self.score(w, self.grid, self.edges, self.states, cur_seq, cur_label, prod, pairwise_prod, flag)
            print 'Loss of sample %d: %f'%(i, logp)
            print 'sample gradient: ', g_tmp
            if mode == DEBUG:
                pdb.set_trace()
            logl += logp + np.log(Z)
            g += g_tmp
            if i % ITER_ROUND == 0: 
                LOG(RELEASE, 'Processing sample %d/%d, l: %f'%(i, sample_len, logl ))
                LOG(RELEASE, g)
        for k in xrange(len(g)):
            g[k] += 2* lambda_* w[k]
        logl += lambda_*sum(w**2)
        print 'Loss function:', logl
        print 'Gradient:', g
        pdb.set_trace()
        return logl,g               

    def trainer(self, trainX, trainY):
        (num_samples, dim) = np.shape(trainX)
        (_, voxel_num) = np.shape(trainY)
        w = np.random.rand(dim*2, 1)
        bounds = [(0,1)]*(dim*2)
        [w,f,d] = op.fmin_l_bfgs_b(self.objective_and_gradients_batch, w, fprime = None, args=(trainX, trainY, 0), bounds=bounds,iprint=100 )
        self.w = w
        pdb.set_trace()
        print 'minimum_function: %f'%f

    def unary_func(self, node, dim, state, value ):
        if node < 0:
            return 0
        return -np.log(self.histo[node][dim][int(state)][self.rss_range(value)])

    def pairwise_func(self, edge, dim, state, value):
        return (1 + np.exp(-( self.histo[edge[0]][dim][int(state[0])][self.rss_range(value)] 
                            - self.histo[edge[1]][dim][int(state[1])][self.rss_range(value)]
                            )**2))/2 

    def get_unary_features(self, data, nodes, labels):
        num_unary_feature = len(data)
        return np.array([ [ self.unary_func(node, j, state, data[j] )
                                        for j in xrange(num_unary_feature) ]
                                        for node, state in zip(nodes, labels)]) 
                             
    def get_pairwise_features(self, data, edges, labels):
        num_pairwise_feature = len(data)
        return np.array([[ self.pairwise_func(edge, j, [labels[edge[0]], labels[edge[1]]], data[j])
                                        if labels[edge[0]] != labels[edge[1]] else 0 
                                        for j in xrange(num_pairwise_feature) ] 
                                        for edge in edges ])


        
    def unary_sum(self, w, data, nodes, labels):
        '''
            w: unary weights
            data: input unary feature of a sample with dimension(num_unary_feature)
            nodes: node in current set
            labels: label corresponding to nodes

            unary_features: unary features in potential function(observation dimension* state )
            
        '''
        _sum = 0
        num_unary_feature = len(data)
        num_nodes = len(nodes)
        #for state in xrange(states):
        #    _sum += sum([ w[state*num_unary_feature + k]* unary_features[i][k] if label[i] == state for k in xrange(num_unary_feature) for i in xrange(num_nodes) ])
        unary_features = self.get_unary_features(data, nodes, labels)
        _sum += sum([ w[k]* unary_features[i][k] for k in xrange(num_unary_feature) for i in xrange(num_nodes) ])
        return _sum

    def pairwise_sum(self, w, data, edges, labels):
        '''
            w: pairwise weights
            data: input pairwise feature of a sample with dimension(num_unary_feature)
            edges: edges involved in current level set.
            labels: label corresponding to nodes

            
        '''
        num_pairwise_feature = len(data) 
        num_edges = len(edges)
        _sum = 0
        #for state1,state2  in zip(xrange(states), xrange(states)):
        #    state_index = state1* states + state2
        #    _sum += sum([ w[state_index * num_pairwise_feature + k]* pairwise_features[i][k] if labels[t[0]] == state1 and labels[t[1]] == state2 for k in xrange(num_pairwise_feature) for i, t in enumerate(edges) ])
        if num_edges > 0:
            pairwise_features = self.get_pairwise_features(data, edges, labels)
            _sum += sum([ w[k]* pairwise_features[i][k] for k in xrange(num_pairwise_feature) for i, t in enumerate(edges) ])
        return _sum

    def get_Td(self, grid, d, edges, pre_nodes):
        cur_nodes, sub_edges_index = [], []
        (M, N) = np.shape(grid)
        start_x = min(d, M)
        x = start_x
        while x >= 1 and d - x + 1 <= N:
            node_id = [x, d + 1 - x]
            x -= 1
            node =  np.where(np.all(self.mapper == node_id, axis = 1))    
            if len(node) != 0:
                cur_nodes.append(node[0][0])
        for i, t in enumerate(edges):
            if (t[0] in pre_nodes and t[1] in cur_nodes) or (t[1] in pre_nodes and t[0] in cur_nodes):
                sub_edges_index.append(i)       
        sub_edges = edges[sub_edges_index]
        return cur_nodes, sub_edges

    def compute_M(self, grid, w, edges, seq,voxel_num, states):
        num_unary_features = len(seq)
        (M, N) = np.shape(grid)
        Md = [] 
        pre_nodes = [-1]
        pre_labels = [0,1]
        for d in xrange(1, M+N+1):
            cur_nodes, sub_edges = self.get_Td(grid, d, edges, pre_nodes)
            if len(cur_nodes) == 0:
                cur_nodes = [-1]
            cur_labels = list(itertools.product([i for i in xrange(states)], repeat=len(cur_nodes))) 
            Md_tmp = np.ones((states ** len(pre_nodes), states ** len(cur_nodes)))
            for j, curl in enumerate(cur_labels):
                u_sum = self.unary_sum(w[0 : num_unary_features], seq, cur_nodes, curl)
                for i, prel in enumerate(pre_labels):
                    label = np.zeros(voxel_num)
                    if d != M + N:
                        label[cur_nodes] = curl
                    if d != 1:
                        label[pre_nodes] = prel
                    #unary_sum = self.unary_sum(w[: num_unary_features* states], unary_features[cur_nodes], states, curl)
                    #pairwise_sum = self.pairwise_sum(w[num_unary_feature,:], sub_edge_features, sub_edges, states, labels)
                    p_sum = self.pairwise_sum(w[num_unary_features:], seq, sub_edges, label)
#print 'd: %d, u_sum: %f, p_sum: %f'%(d, u_sum, p_sum)
                    Md_tmp[i][j] = np.exp(-(u_sum + p_sum))
            Md.append(np.array(Md_tmp)) 
            pre_nodes = cur_nodes
            pre_labels = cur_labels
        return Md

    def compute_alpha_score(self, grid, states, Md):
        (M, N) = np.shape(grid)
        alpha = []
        # initial
        alpha.append([1,0])
        for d in xrange(1, M + N + 1):
            alpha.append(np.dot( alpha[d-1], Md[d-1]) )
        return alpha

    def compute_beta_score(self, grid, states, Md):
        (M, N) = np.shape(grid)
        beta, reverse_beta = [], []
        # initial
        beta.append([1,0])
        for d in xrange(1, M + N + 1 ):
            beta.append(np.transpose(np.dot(Md[M + N - d], beta[d - 1])))
        for b in reversed(beta):
            reverse_beta.append(b)
        return reverse_beta

    def compute_marginal(self, grid, edges, states, Z, alpha, beta, Md):
        (M, N) = np.shape(grid)
        num_edges = len(edges)
        prod = np.zeros((states, M*N) )
        pairwise_prod = np.zeros((states** 2, num_edges ))
        pre_nodes = [-1]
        pre_labels = [-1]
        for d in xrange(1, M + N):
            cur_nodes, sub_edges = self.get_Td(grid,d, edges, pre_nodes)
            cur_labels = list(itertools.product([i for i in xrange(states)], repeat=len(cur_nodes))) 
            # marginal for unary
            for i, node in enumerate(cur_nodes):
                for state in xrange(states):
                    ci = [ j for j, labels in enumerate(cur_labels) if labels[i] == state]
                    prod[state][node] = sum( np.multiply(alpha[d][ci], beta[d][ci]) )/Z
            for si, sub_edge in enumerate(sub_edges):
                (s, t) = sub_edge
                for state_id in xrange(states**2):
                    state1 = state_id/states
                    state2 = state_id%states
                    edge_id = np.where(np.all(edges == sub_edge, axis = 1))[0][0]    
                    if s in cur_nodes:
                        ni = np.where(cur_nodes == s)[0][0]
                        pni = np.where(pre_nodes == t)[0][0]    
                    else:
                        ni = np.where(cur_nodes == t)[0][0]
                        pni = np.where(pre_nodes == s)[0][0]    
                        ci = [j for j, curl in enumerate(cur_labels) if curl[ni] == state1 ]
                        pi = [j for j, prel in enumerate(pre_labels) if prel[pni] == state2 ]
                        # sum over alpha_{d_2}(T'_{d-1}|x)*Md(T'_{d-1}, T_d | x)*beta_d(Td|x)
                        pairwise_prod[state_id][edge_id] = sum(sum(np.einsum('i,ij,j->ij',alpha[d-1][pi], Md[d-1][np.ix_(pi,ci)], beta[d][ci])))/Z
            pre_nodes = cur_nodes
            pre_labels = cur_labels 
        return prod, pairwise_prod 

    def vector_sum(self, vec):
        if not isinstance(vec, list) and not isinstance(vec, np.ndarray):
            return vec
        while isinstance(vec,list) or isinstance(vec, np.ndarray):
            vec = sum(vec)
        return vec

    def score(self, w, grid, edges, states, seq, label, prod, pairwise_prod, flag):
        # flag : 0: both of gradient and objective function
        #        1: gradient only
        #        2: objective function only
        (M, N) = np.shape(grid)
        num_unary_features, num_pairwise_features = len(seq), len(seq)
        logp = 0
        pre_nodes = [-1]
        pre_labels = [0]
        g = np.zeros(np.shape(w))
        for d in xrange(1, M+N):
            cur_nodes, sub_edges = self.get_Td(grid, d, edges, pre_nodes)
            cur_labels = list(itertools.product([i for i in xrange(states)], repeat=len(cur_nodes))) 
            ## empirical labels
            em_unary_features = self.get_unary_features(seq, cur_nodes, label)
            em_pairwise_features = self.get_pairwise_features(seq, sub_edges, label)

            logp += sum([ w[k]* em_unary_features[i][k] for k in xrange(num_unary_features) for i,t in enumerate(cur_nodes) ])
            if len(sub_edges) > 0:
                logp += sum([ w[k + num_unary_features]* em_pairwise_features[i][k] for k in xrange(num_pairwise_features) for i, t in enumerate(sub_edges) ])
    
            ## probability labels
            if flag != 2:
                for k in xrange(num_unary_features):
                    g[k] += sum( em_unary_features[:,k] ) - sum([ (self.get_unary_features(seq, [node], [state]))[0][k]* prod[state][node] for node in cur_nodes for state in xrange(states) ])   
                pb_label = list(label)
                for k in xrange(num_pairwise_features):
                    if len(sub_edges) == 0:
                        break 
                    tmp = self.vector_sum(em_pairwise_features[:,k]) 
                    for edge in sub_edges:
#LOG(DEBUG, 'edge [%d, %d]'%(edge[0], edge[1]))
                        for state_id in xrange(states**2):
                            state1 = state_id / states
                            state2 = state_id % states
                        
                            pb_label[edge[0]] = state1
                            pb_label[edge[1]] = state2
                            cur_f = self.get_pairwise_features(seq, [edge], pb_label)
                            edge_id = np.where(np.all(edges == edge, axis = 1))[0][0]    
                            tmp -= cur_f[0][k]* pairwise_prod[state_id][edge_id]
                    g[k + num_unary_features] += tmp
#                    LOG(DEBUG, 'feature %d, gradient %lf'%(k,tmp))
            pre_nodes = cur_nodes
            pre_labels = cur_labels
        return g, logp

    def predict(self, testX):
        y_pred = self.map_estimate(testX)
        return y_pred
 
    def map_estimate(self, seq):
        (sample_len, dim) = np.shape(seq)
        dataX = []
        edge_dict = {}
        max_iter = 100
        for i, edge in enumerate(self.edges):
            if edge[0] not in edge_dict.keys():
               edge_dict[ edge[0] ] = [(i, edge)]
            else:
               edge_dict[ edge[0] ] = edge_dict[ edge[0] ] + [(i, edge)]
                   
        for i in xrange(sample_len):
            labels = np.array([ int(np.random.random()*self.states) for k in xrange(self.n_nodes)]) 
            # build matrix D
            D = np.array([[ np.squeeze(self.get_unary_features(seq[i], [node] , [label])) for label in xrange(self.states)]for node in xrange(self.n_nodes)])
            # build matrix v
            v = np.array([[ [ [ self.pairwise_func(t, j, [l,k], seq[i][j])
            for j in xrange(dim)]
                    for l in xrange(self.states)] 
            for k in xrange(self.states) ]
            for t in self.edges])
            labels = self.cv_alpha_extension(D, self.w, max_iter, edge_dict, v, labels)
            print labels
            pdb.set_trace()

    def cv_alpha_extension(self, D, w, max_iter, edge_dict, v, labels):
        '''
            D: unary cost for each label for different nodes, shape of (num_nodes,num_labels, num_features)
            edge_dict { key:id of first_node, value: (id in edge feature array 'v', list of edges starting with first_node) }
            v: edge feature array, shape of ( id, n_states, n_states features, )
        '''
        assert labels.shape[0] == D.shape[0]
        assert D.shape[1] == self.states

        n_nodes = labels.shape[0] 
        num_unary_feature = D.shape[2]
        num_pairwise_feature = v.shape[3]
        success = 0
        best_energy = sys.maxint
        for it in xrange(max_iter):
#print 'Iteration %d'%it
            # Process the neighbors
            for alpha in xrange(self.states):
                print 'alpha: %d'%alpha
                # create the graph
                g = maxflow.GraphFloat()
                g.add_nodes(n_nodes)

                for node_index in xrange(n_nodes):
#                   print "node: ", node_index
                    label = labels[node_index]
                    # TBD
                    t1 = self.vector_sum( np.multiply( w[:num_unary_feature],D[node_index][alpha] ))
                    t2 = sys.maxint
                    if label != alpha:
                        t2 = self.vector_sum( np.multiply( w[:num_unary_feature],D[node_index][label] ))
#print t1, t2
                    g.add_tedge(node_index, t1, t2)
                    if node_index not in edge_dict.keys():
                        continue
                        
                    for (_id, edge) in edge_dict[node_index]:
                        assert edge[0] == node_index
                        nnode_index = edge[1]
                        nlabel = labels[nnode_index]
                        nnstates = self.states ** 2 
                        dist_label_alpha = sum([ np.mean([w[num_unary_feature:]*v[_id][label][alpha],w[ num_unary_feature:]*v[_id][alpha][nlabel]])] )

                        if label == nlabel:
                            g.add_edge(node_index, nnode_index, dist_label_alpha, dist_label_alpha)
                            continue
                        # if labelk are different , add an extra node
                        dist_nlabel_alpha = sum([ np.mean([w[num_unary_feature:]*v[_id][nlabel][alpha],w[ num_unary_feature:]*v[_id][alpha][nlabel]])])
                        dist_label_nlabel =  sum([ np.mean([w[num_unary_feature:]*v[_id][label][nlabel],w[ num_unary_feature:]*v[_id][nlabel][label]])])
                        assert(dist_label_alpha + dist_nlabel_alpha >= dist_label_nlabel)
                        extra_node = g.add_nodes(1)
                        g.add_tedge(extra_node, 0, dist_label_nlabel)
                        g.add_edge(node_index, extra_node, dist_label_alpha, dist_label_alpha)
                        g.add_edge(nnode_index, extra_node, dist_nlabel_alpha, dist_nlabel_alpha)
#                       print "add a new node, with tweight %f, weight to two end notes(%f, %f)"%(dist_label_nlabel, dist_label_alpha, dist_nlabel_alpha)

                energy = g.maxflow()
                for node_index in xrange(n_nodes):
                    if g.get_segment(node_index) == 1: # SINK = 1, SOUCE = 0
                        labels[node_index] = alpha

                print energy, labels 
                if energy < best_energy:
                    success = 1
                    best_energy = energy
                    print "Energy of the last cut (alpha = %r): %r"%(alpha, energy)

                if not success:
                    print 'failed!'
                    break
        return labels

               


def pickle_save(fname, data):
    with open(fname, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
        print "saved to %s"%fname

def pickle_load(fname):
    with open(fname, 'rb') as _input:
        return pickle.load(_input)

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray):
            dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem
    return dict

def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays 
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []            
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list 


def LOG(log_level, mesg):
    if log_level == DEBUG and mode == DEBUG:
        print mesg

if __name__ == '__main__':
        
    file_p = 'data/mat/'
    samples = loadmat(file_p + 'sample.mat')['Train']
    config = loadmat(file_p + 'config.mat')['Config']
    test = loadmat(file_p + 'Test_fg_three_13_17_112.mat')['Test']
    model_file = 'model.pkl'
    if oper == 'train':
        crf_instance = CRF_SOLVER(config)
        trainX = np.array(samples['seq'])
        tmpY = np.array(samples['label'])
        voxel_num = len(crf_instance.mapper)
        trainY = np.zeros((len(tmpY), voxel_num))
        for i, y in enumerate(tmpY):
            trainY[i][y-1] = 1
        crf_instance.trainer(trainX, trainY)
        pickle_save(model_file,crf_instance)
        pdb.set_trace()

    elif oper == 'test':
        crf_instance = pickle_load(model_file)
#crf_instance.w = [-0.82378552, -1.16889145, -0.80623083,  0.50265313,  0.50306962, 0.50292155]
#        (crf_instance.n_nodes, _) = np.shape(crf_instance.mapper)
        testX = np.array(test['seq']);
        print crf_instance.predict(testX)
        pdb.set_trace()
    else:
        print 'Unsupported Operation.'
        sys.exit()

#labels = crf_instance.predict(test_case, 1)
#crf_instance.score(trainX, trainY)
#print crf_instance.predict(trainX)

