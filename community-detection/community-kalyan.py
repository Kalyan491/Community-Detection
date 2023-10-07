import numpy as np
import scipy.linalg
import time
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import seaborn as sb

if not os.path.exists('./plots/'):
    os.makedirs('./plots/')
class Louvain_Algorithm():
    def __init__(self,nodes_connectivity_list) -> None:
        self.Graph=nx.Graph()
        self.nodes_connectivity_list=np.unique(nodes_connectivity_list,axis=0)
        self.number_of_edges=len(self.nodes_connectivity_list)
        self.neighbours = []
        self.m=self.number_of_edges
        self.constant=2 * self.m
        self.Graph.add_edges_from(nodes_connectivity_list)
        self.nodes = self.Graph.nodes()
        self.number_of_nodes=len(self.nodes)
        self.dataset_name="bitcoin"
        if self.number_of_nodes==4039:
            self.dataset_name="facebook"
        self.Adjacency_matrix=self.get_Adjacency_matrix_scaled(self.nodes_connectivity_list,len(self.nodes))
        self.Degree_matrix=self.get_degree_matrix(self.Adjacency_matrix)
        self.Adjacency_matrix_plot=self.get_Adjacency_matrix_plot(self.nodes_connectivity_list,len(self.nodes))
        self.lovian_communities = np.arange(self.number_of_nodes)
        for node in range(len(self.nodes)):
            self.neighbours.append(list(self.Graph.neighbors(node)))

    def get_Adjacency_matrix_scaled(self,nodes_connectivity_list_fb,Node_count):
        A=np.zeros((Node_count,Node_count))
        for each in nodes_connectivity_list_fb:
            A[each[0]][each[1]]=1
            A[each[1]][each[0]]=1
        return A/self.constant
    
    def get_Adjacency_matrix_plot(self,nodes_connectivity_list_fb,Node_count):
        A=np.zeros((Node_count,Node_count))
        for each in nodes_connectivity_list_fb:
            A[each[0]][each[1]]=1
            A[each[1]][each[0]]=1
        return A
    
    def plot_adjacency_matrix_sorted(self,graph_partition):
        fig = plt.figure(figsize=(5, 5)) # in inches
        graph_partition=np.array(graph_partition)
        #print(graph_partition[:, 1])
        sorted_indices = np.argsort(graph_partition[:, 1])

    # Sort the array based on the sorted indices
        graph_partition = graph_partition[sorted_indices]

        A_new=self.Adjacency_matrix_plot[graph_partition[:,0]][:,graph_partition[:,0]]
        plt.figure()
        plt.spy(A_new)
        if self.dataset_name=="facebook":
            plt.savefig("./plots/"+"facebook_Adj_matrix_lovian.png")
        else:
            plt.savefig("./plots/"+"bitcoin_Adj_matrix_lovian..png")

    def get_degree_matrix(self,A):
        # D=np.zeros(A.shape, dtype=int)
        # for i in range(len(D)):
        #     D[i][i]=np.sum(A[i])
        D = np.sum(A, axis = 1)
        return D
    
    def plot_graph_vis(self,graph_partition):
        G = nx.Graph(self.Adjacency_matrix_plot)
        pos = nx.spring_layout(G)
        graph_partition=np.array(graph_partition)
        node_colors = graph_partition[:, 1]
        plt.figure()
        nx.draw(G, pos, node_color=node_colors,cmap=plt.cm.Set1, with_labels=False)
        if self.dataset_name=="facebook":
            plt.savefig("./plots/"+'louvain_facebook_plot.png')
        else:
            plt.savefig("./plots/"+'louvain_bitcoin_plot.png')


    def get_modularity(self):
        communities = np.unique(self.lovian_communities)
        mod=0
        for comm in communities:
            curr_comm = np.where(self.lovian_communities == comm)[0]
            sigma_total = sum(self.Degree_matrix[node] for node in curr_comm)
            sigma_in = np.sum(self.Adjacency_matrix[curr_comm][:,curr_comm])
            mod += sigma_in - sigma_total**2
        return mod


    def get_sigma_in(self,community_nodes):
        return np.sum(self.Adjacency_matrix[community_nodes][:, community_nodes])

    def get_q_merge(self,moving_community,node):
        k_i = self.Degree_matrix[node]
        moving_community_nodes = np.where(self.lovian_communities == moving_community)[0]
        sigma_t=0
        #for node in moving_community_nodes:
            # sigma_t += self.Degree_matrix[node][node] 
        sigma_t = sum(self.Degree_matrix[node] for node in moving_community_nodes)

        #print(sigma_in,sigma_t)
        k_i_in = 2*np.sum(self.Adjacency_matrix[node,moving_community_nodes])

        return k_i_in - 2*sigma_t*k_i

    def get_q_demerge(self,node):
        k_i = self.Degree_matrix[node]
        current_community = self.lovian_communities[node]
        current_community_nodes = np.where(self.lovian_communities == current_community)[0]
        sigma_t=0
        #for node in current_community_nodes:
            # sigma_t += self.Degree_matrix[node][node]
        sigma_t = sum(self.Degree_matrix[node] for node in current_community_nodes)
        k_i_in = 2*np.sum(self.Adjacency_matrix[node,current_community_nodes])
        # q_after = (sigma_in - k_i_in)/(self.constant) - ((sigma_t - k_i)/(self.constant))**2 - ((k_i/self.constant)**2)- (sigma_in/(self.constant)) - (sigma_t/self.constant)**2
        return 2*k_i*sigma_t - 2*k_i**2 - k_i_in



    def get_best_community(self,node,neighbour_communities,maximum_q,q_demerging_value,best_community):
        for neighbor_comm in neighbour_communities:
            if neighbor_comm == self.lovian_communities[node]:
                continue
            q_merging = self.get_q_merge(neighbor_comm,node)
            delta_Q = q_demerging_value + q_merging
            #print(q_demerging_value, q_merging,   delta_Q)
            if delta_Q > maximum_q:
                best_community = neighbor_comm
                maximum_q= delta_Q
        return best_community,maximum_q     

    def Run_LA_Phase1(self):
        while True:
            changes=0
            has_updated=False
            for node in range(len(self.nodes)):
                best_community = self.lovian_communities[node]
                neighbour_communities = np.unique(self.lovian_communities[self.neighbours[node]])
                maximum_q = 0
                q_demerging = self.get_q_demerge(node)
                # print(q_demerging)
                best_community,max_q=self.get_best_community(node,neighbour_communities,maximum_q,q_demerging,best_community)

                # print(max_q)

                if max_q>0 and best_community != self.lovian_communities[node]:
                    #print(self.lovian_communities[node],best_community)
                    has_updated=True
                    self.lovian_communities[node] = best_community
                    changes+=1
            
            #print("No..of changes:",changes,"No.of Communities:",len(np.unique(self.lovian_communities)),"Modularity:",self.get_modularity())

            if has_updated == False:
                break
        #print(self.lovian_communities)
        graph_partition=[[n,comm] for n,comm in enumerate(self.lovian_communities)]
        return graph_partition

def import_facebook_data(data_path):
        with open(data_path) as f:
            lines = f.readlines()
        edge = []
        for each in lines:
            row = each.split(' ')
            edge.append(row)
        edge=np.unique(edge,axis=0)
        return np.array(edge).astype(int)
            

def import_bitcoin_data(data_path):
    df=pd.read_csv(data_path,header=None)
    #print(df.head())
    df_new=df.iloc[:, :2]
    nodes=np.unique(df_new).astype(int)
    map_dict={}
    for i in range(len(nodes)):
        map_dict[nodes[i]]=i
    df_new_list=df_new.values.tolist()
    edges=set()
    df_final=[]
    for edge in df_new_list:
        edge[0],edge[1]=map_dict[edge[0]],map_dict[edge[1]]
        edge=sorted(edge)
        if tuple(edge) not in edges:
            edges.add(tuple(edge))
            df_final.append(edge)
    return df_final
            


def get_unique_from_column(data,col):

    # Create an empty set to store unique values
    unique_values = set()

    # Iterate through the list of lists and add the second column values to the set
    for row in data:
        unique_values.add(row[1])

    # Convert the set back to a list if needed
    unique_values_list = list(unique_values)
    return unique_values_list

def get_Number_of_nodes(nodes_connectivity_list):
    return len(np.unique(nodes_connectivity_list))


def get_Adjacency_list(A):
    #A=get_Adjacency_matrix(nodes_connectivity_list_fb)
    num_nodes = len(A)
    adjacency_list = {}

    for i in range(num_nodes):
        adjacency_list[i] = []
        for j in range(num_nodes):
            if A[i][j] == 1:
                adjacency_list[i].append(j)

    return adjacency_list

def get_Adjacency_matrix(nodes_connectivity_list, Adjacency_map,Node_count):
    A=np.zeros((Node_count,Node_count))
    for each in nodes_connectivity_list:
        i=Adjacency_map[each[0]]
        j=Adjacency_map[each[1]]
        A[i][j]=1
        A[j][i]=1
    return A

def get_Adjacency_matrix_plot(nodes_connectivity_list):
    Node_count=len(np.unique(nodes_connectivity_list))
    A=np.zeros((Node_count,Node_count))
    for each in nodes_connectivity_list:
        i=each[0]
        j=each[1]
        A[i][j]=1
        A[j][i]=1
    return A

def get_degree_matrix(A):
    D=np.zeros(A.shape)
    for i in range(len(D)):
        D[i][i]=np.sum(A[i])
    return D

def plot_fielder_vector(feilder_vector,dataset_name):
    x=np.arange(feilder_vector.shape[0])
    y=np.sort(feilder_vector)
    plt.figure()
    plt.scatter(x, y, label='Data Points', color='blue', marker='o')
    if dataset_name=="Facebook":
        plt.savefig("./plots/"+"Fielder_Vector_Facebook.png")
    else:
        plt.savefig("./plots/"+"Fielder_Vector_bitcoin.png")

def plot_adjacency_matrix_one_iter(nodes_list,fielder_vector,dataset_name):
    A=get_Adjacency_matrix_plot(nodes_list)
    #K=sb.heatmap(A)
    sorted_fielder_fb=np.argsort(fielder_vector,axis=0).reshape(-1)
    #print(sorted_fielder_fb)
    A_new = deepcopy(A)
    A_new = np.take(A_new, sorted_fielder_fb, axis = 0)
    A_new = np.take(A_new, sorted_fielder_fb, axis = 1)
    plt.figure()
    plt.spy(A_new)
    if dataset_name=="Facebook":
        plt.savefig("./plots/"+"Facebook_Adj_matrix_spectral_sorted_one_iter.png")
    else:
        plt.savefig("./plots/"+"Bitcoin_Adj_matrix_spectral_sorted_one_iter.png")

def plot_adjacency_matrix(graph_partition,nodes_connectivity_list):
    A=get_Adjacency_matrix_plot(nodes_connectivity_list)
    #print(A.shape)
    sorted_indices = np.argsort(graph_partition[:, 1])

    # Sort the array based on the sorted indices
    sorted_g = graph_partition[:,0][sorted_indices]

    # A_new=A[graph_partition[:,0]][:,graph_partition[:,0]]
    A_new = deepcopy(A)
    A_new = np.take(A_new, sorted_g, axis = 0)
    A_new = np.take(A_new, sorted_g, axis = 1)
    # fig = plt.figure(figsize=(5, 5)) # in inches
    # plt.imshow(A_new,
    #             cmap="inferno",
    #             interpolation="none")
    plt.figure()
    plt.spy(A_new)
    if A.shape[0]==4039:
        plt.savefig("./plots/"+"Facebook_Adj_matrix_spectral_sorted.png")
    else:
        plt.savefig("./plots/"+"Bitcoin_Adj_matrix_spectral_sorted.png")
    return A_new

def plot_graph_vis(A,graph_partition):
    A=np.array(A)
    G = nx.Graph(A)
    pos = nx.spring_layout(G)
  
    #print(com_0)
    plt.figure()
    node_colors = graph_partition[:, 1]
    nx.draw(G, pos, node_color=node_colors, cmap=plt.cm.Set1,with_labels=False)

    if A.shape[0]==4039:
        plt.savefig("./plots/"+"Facebook_Graph_visualization_spectral.png")
    else:
        plt.savefig("./plots/"+"Bitcoin_Graph_visualization_spectral.png")


def plot_graph_vis_one_iter(A,graph_partition,dataset_name):
    A=np.array(A)
    #print(A.shape)
    G = nx.Graph(A)
    pos = nx.spring_layout(G)
  
    #print(com_0)
    node_colors = graph_partition[:, 1]
    plt.figure()
    nx.draw(G, pos, node_color=node_colors, cmap=plt.cm.Set1,with_labels=False)
    if dataset_name=="Facebook":
        plt.savefig("./plots/"+"Facebook_Graph_visualization_spectral_one_iter.png")
    else:
        plt.savefig("./plots/"+"Bitcoin_Graph_visualization_spectral_sorted_one_iter.png")

def get_nodes(nodes_connectivity_list):
    nodes=set()
    for each in nodes_connectivity_list:
        nodes.add(each[0])
        nodes.add(each[1])
    return sorted(list(nodes))

def spectralDecomp_OneIter(nodes_connectivity_list):
     
    selected_nodes=np.unique(nodes_connectivity_list).reshape(-1)
    
    Adjacency_map={}
    
    count=0
    for each in selected_nodes:
        Adjacency_map[each]=count
        count+=1

    Adjacency_matrix=get_Adjacency_matrix(nodes_connectivity_list,Adjacency_map,len(selected_nodes))
    
    Degree_Matrix=get_degree_matrix(Adjacency_matrix)
    Laplacian_matrix=Degree_Matrix-Adjacency_matrix
    #For Normalized_cut
    #print('lol ',Laplacian_matrix.shape, Degree_Matrix.shape)
    eigvals, eigvecs = scipy.linalg.eigh(Laplacian_matrix, Degree_Matrix)
    #For mincut(Use the below)
    #eigvals, eigvecs = scipy.linalg.eigh(Laplacian_matrix)

    sorted_indices = np.argsort(eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Get the second largest eigenvector
    second_smallest_eigenvector = eigvecs[:, 1]
    community_assignment = second_smallest_eigenvector < 0  # True for nodes in the first community, False otherwise
    if(len(np.unique(community_assignment))!=2):
        return None,None,None
    
    # Extract the indices of nodes in the first and second communities
    community_index_0 = selected_nodes[np.where(community_assignment)[0][0]]
    community_index_1 = selected_nodes[np.where(~community_assignment)[0][0]]
    comm_heads=[]
    for each in second_smallest_eigenvector:
        if each>=0:
            comm_heads.append(community_index_1)
        else:
            comm_heads.append(community_index_0)

    #print(community_index_0,community_index_1)
    comm_heads=np.array(comm_heads)

    graph_partition=np.column_stack((selected_nodes,comm_heads))

    # Sort the eigenvalues and eigenvec

    #print("partition:", graph_partition_fb)
    graph_partition = np.array(graph_partition).astype(int)
    #print(np.unique(graph_partition[:,1]),"kal")
    fielder_vec_fb=second_smallest_eigenvector
    
    #print(fielder_vec_fb)
    #plot_graph_vis(Adjacency_matrix,graph_partition)
    #print(community_index_0,community_index_1,"op")
    #print(c_0,c_1,"qqq")

    return fielder_vec_fb, Adjacency_matrix, graph_partition

def plot_function_one_iter(fielder_vec,nodes_connectivity_list,graph_partition,dataset_name):
    plot_fielder_vector(fielder_vec,dataset_name)
    plot_adjacency_matrix_one_iter(nodes_connectivity_list,fielder_vec,dataset_name)
    plot_graph_vis_one_iter(get_Adjacency_matrix_plot(nodes_connectivity_list),graph_partition,dataset_name)


def get_relevant_edges(nodes,edge_list):
    new_list=[]
    for each in edge_list:
        if each[0] in nodes and each[1] in nodes:
            new_list.append(each)
    return new_list

def get_graph_partition(nodes_connectivity_list):
    _,_,graph_partition=spectralDecomp_OneIter(nodes_connectivity_list)
    return graph_partition

def spectralDecomposition_multi_iter(nodes_connectivity_list):
    #print(depth)

        fielder_vector,Adjacency_matrix,graph_partition=spectralDecomp_OneIter(nodes_connectivity_list)
        if fielder_vector is None:
            return []
         # True for nodes in the first community, False otherwise
        #print(len(np.unique(community_assignment)))
        
        sorted_fielder_vector=np.sort(fielder_vector)
        #print(sorted_fielder_vector)
        diff_vector=[]
        for i in range(len(sorted_fielder_vector)-1):
            diff_vector.append(sorted_fielder_vector[i+1]-sorted_fielder_vector[i])

        diff_vector=np.array(diff_vector)
        std=np.std(diff_vector)
        mean=np.mean(diff_vector)
        diff_value_max=np.max(diff_vector)

        beta = 200
        #print(diff_value_max,std)
        if(diff_value_max<(beta*mean)):
            return graph_partition

        #graph_partition=get_graph_partition(nodes_connectivity_list)
        current_coms=np.unique(graph_partition[:,1])
        #print("Current Partitioned Communities:",current_coms)
        new_nodes_list1=[]
        new_nodes_list2=[]
        for i in graph_partition:
            if(i[1]==current_coms[0]):
                new_nodes_list1.append(i[0])
            if(i[1]==current_coms[1]):
                new_nodes_list2.append(i[0])
        new_edges_list1=[]
        new_edges_list2=[]
        
        # print(nodes_connectivity_list)
        new_edges_list1=[[node1,node2] for node1,node2 in nodes_connectivity_list if node1 in new_nodes_list1 and node2 in new_nodes_list1]
        new_edges_list2=[[node1,node2] for node1,node2 in nodes_connectivity_list if node1 in new_nodes_list2 and node2 in new_nodes_list2]
        #print(len(new_edges_list1),len(new_edges_list2))
        division1 = []
        division2 = []
        if(len(new_edges_list1) != 0 and len(new_edges_list2)!= 0):
            division1=spectralDecomposition_multi_iter(new_edges_list1)
            division2=spectralDecomposition_multi_iter(new_edges_list2)
        if len(division1) == 0  or len(division2) == 0:
            return graph_partition
        
        g=np.row_stack((division1,division2))
        before_split=np.unique(nodes_connectivity_list).reshape(-1)
        after_split_1=np.unique(new_edges_list1).reshape(-1)
        after_split_2=np.unique(new_edges_list2).reshape(-1)

        for node in before_split:
            if node not in after_split_1 and node not in after_split_2:
                index=np.where(graph_partition[:,0]==node)[0]
                g=np.row_stack((g,graph_partition[index]))
        sorted_indices = np.argsort(g[:, 0])

        # Sort the array based on the sorted indices
        sorted_graph_partition = g[sorted_indices]

        return sorted_graph_partition


def spectralDecomposition(nodes_connectivity_list):
    start_time = time.time()
    sorted_graph_partition=spectralDecomposition_multi_iter(nodes_connectivity_list)
    end_time = time.time()
    execution_time_spectral = end_time - start_time

    print("Time taken in seconds For Spectral Decomposition:",execution_time_spectral)

    print("No.of Communities:",len(np.unique(sorted_graph_partition[:,1])))
    print("Final Community ids:",(np.unique(sorted_graph_partition[:,1])))

    A=get_Adjacency_matrix_plot(nodes_connectivity_list)

    #print(len(sorted_graph_partition))

    plot_graph_vis(A,sorted_graph_partition)

    return sorted_graph_partition

def createSortedAdjMat(graph_partition, nodes_connectivity_list):
    sorted_Adj = plot_adjacency_matrix(graph_partition,nodes_connectivity_list)
    return sorted_Adj

def final_graph_partition_modularity(edges,graph_partition):
    G = nx.Graph()
    G.add_edges_from(edges)
    unique_id = np.unique(np.array(graph_partition)[:,1])
    communities = []
    for i in unique_id:
        C = set(np.array(graph_partition)[:,0][np.where(np.array(graph_partition)[:,1] == i)[0]])
        communities.append(C)
    Q = nx.community.modularity(G, communities)
    return Q



def louvain_one_iter(nodes_connectivity_list):

    Louvian=Louvain_Algorithm(nodes_connectivity_list)
    start_time = time.time()
    graph_partition=Louvian.Run_LA_Phase1()
    end_time = time.time()
    if(Louvian.dataset_name=="facebook"):
        execution_time_fb_louvain = end_time - start_time
        print("Time Taken in seconds For Louvain Algorithm(Facebook):",execution_time_fb_louvain)
    else:
        execution_time_btc_louvain = end_time - start_time
        print("Time Taken in seconds For Louvain Algorithm(Bitcoin):",execution_time_btc_louvain)
    Louvian.plot_adjacency_matrix_sorted(graph_partition)
    Louvian.plot_graph_vis(graph_partition)

    print("No.of Communities:",len(np.unique(np.array(graph_partition)[:,1])))
    print("Final Community ids:",(np.unique(np.array(graph_partition)[:,1])))
    return graph_partition

if __name__ == "__main__":
    print("-------------------DATASET-1:Facebook---------------------")
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    print("---Part-1:Spectral Decompostion---")

    fielder_vec_fb, adj_mat_fb, graph_partition_fb= spectralDecomp_OneIter(nodes_connectivity_list_fb)

    plot_function_one_iter(fielder_vec_fb,nodes_connectivity_list_fb,graph_partition_fb,"Facebook")

    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)

    print("Modularity of Final Spectral Decomposition partition:",final_graph_partition_modularity(nodes_connectivity_list_fb,graph_partition_fb))

    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)


    print("---Part-2:Louvain Algorithm---")

    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)

    print("Modularity of Final Louvain partition:",final_graph_partition_modularity(nodes_connectivity_list_fb,graph_partition_louvain_fb))


    print("-------------------DATASET-2:Bitcoin----------------------")
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    print("---Part-1:Spectral Decomposition---")
    fielder_vec_btc, adj_mat_btc, graph_partition_btc= spectralDecomp_OneIter(nodes_connectivity_list_btc)

    plot_function_one_iter(fielder_vec_btc,nodes_connectivity_list_btc,graph_partition_btc,"Bitcoin")


    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    print("Modularity of Final Spectral Decomposition partition:",final_graph_partition_modularity(nodes_connectivity_list_btc,graph_partition_btc))

    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    print("---Part-2:Louvain Algorithm----")

    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)


    print("Modularity of Final Louvain partition:",final_graph_partition_modularity(nodes_connectivity_list_btc,graph_partition_louvain_btc))
