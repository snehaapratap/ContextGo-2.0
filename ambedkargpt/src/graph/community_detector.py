from typing import List, Dict, Optional, Set
import networkx as nx
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logger.warning("leidenalg not available, will use Louvain algorithm")

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logger.warning("python-louvain not available")


class CommunityDetector:
    def __init__(self, algorithm: str = "leiden", resolution: float = 1.0):
        self.algorithm = algorithm.lower()
        self.resolution = resolution
        self.communities = {}
        self.node_to_community = {}
        if self.algorithm == "leiden" and not LEIDEN_AVAILABLE:
            logger.warning("Leiden algorithm not available, falling back to Louvain")
            self.algorithm = "louvain"
        
        if self.algorithm == "louvain" and not LOUVAIN_AVAILABLE:
            logger.warning("Louvain algorithm not available, will use connected components")
            self.algorithm = "components"
        
        logger.info(f"CommunityDetector initialized with algorithm={self.algorithm}")
    
    def detect_communities(self, graph: nx.Graph) -> Dict[int, Set[str]]:
        if graph.number_of_nodes() == 0:
            return {}
        
        logger.info(f"Detecting communities using {self.algorithm} algorithm")
        
        if self.algorithm == "leiden":
            communities = self._detect_leiden(graph)
        elif self.algorithm == "louvain":
            communities = self._detect_louvain(graph)
        else:
            communities = self._detect_components(graph)
        
        self.communities = communities
        self._build_node_to_community_mapping()
        
        logger.info(f"Detected {len(communities)} communities")
        return communities
    
    def _detect_leiden(self, graph: nx.Graph) -> Dict[int, Set[str]]:
        node_list = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
        
        ig_graph = ig.Graph(n=len(node_list), edges=edges)
        try:
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self.resolution
            )
        except TypeError:
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition
            )
        communities = defaultdict(set)
        for node_idx, community_id in enumerate(partition.membership):
            communities[community_id].add(node_list[node_idx])
        
        return dict(communities)
    
    def _detect_louvain(self, graph: nx.Graph) -> Dict[int, Set[str]]:
        partition = community_louvain.best_partition(
            graph,
            resolution=self.resolution
        )
        
        communities = defaultdict(set)
        for node, community_id in partition.items():
            communities[community_id].add(node)
        
        return dict(communities)
    
    def _detect_components(self, graph: nx.Graph) -> Dict[int, Set[str]]:
        communities = {}
        for i, component in enumerate(nx.connected_components(graph)):
            communities[i] = component
        return communities
    
    def _build_node_to_community_mapping(self):
        self.node_to_community = {}
        for comm_id, nodes in self.communities.items():
            for node in nodes:
                self.node_to_community[node] = comm_id
    
    def get_community(self, node: str) -> Optional[int]:
        return self.node_to_community.get(node.lower().strip())
    
    def get_community_members(self, community_id: int) -> Set[str]:
        return self.communities.get(community_id, set())
    
    def get_community_subgraph(self, graph: nx.Graph, community_id: int) -> nx.Graph:
        nodes = self.communities.get(community_id, set())
        return graph.subgraph(nodes)
    
    def get_community_info(self, graph: nx.Graph, community_id: int) -> Dict:
        nodes = self.communities.get(community_id, set())
        if not nodes:
            return {}
        
        subgraph = graph.subgraph(nodes)
        node_details = []
        for node in nodes:
            node_data = graph.nodes.get(node, {})
            node_details.append({
                'id': node,
                'text': node_data.get('text', node),
                'type': node_data.get('type', 'UNKNOWN'),
                'descriptions': node_data.get('descriptions', [])
            })

        edge_details = []
        for u, v, data in subgraph.edges(data=True):
            edge_details.append({
                'source': u,
                'target': v,
                'relations': data.get('relations', []),
                'weight': data.get('weight', 1.0)
            })
        
        return {
            'community_id': community_id,
            'size': len(nodes),
            'nodes': node_details,
            'edges': edge_details,
            'density': nx.density(subgraph) if len(nodes) > 1 else 0
        }
    
    def get_all_community_info(self, graph: nx.Graph) -> List[Dict]:
        return [
            self.get_community_info(graph, comm_id)
            for comm_id in sorted(self.communities.keys())
        ]
    
    def get_community_stats(self) -> Dict:
        if not self.communities:
            return {'num_communities': 0}
        
        sizes = [len(nodes) for nodes in self.communities.values()]
        
        return {
            'num_communities': len(self.communities),
            'min_size': min(sizes),
            'max_size': max(sizes),
            'avg_size': np.mean(sizes),
            'total_nodes': sum(sizes)
        }
    
    def merge_small_communities(self, graph: nx.Graph, min_size: int = 2):
        small_communities = [
            comm_id for comm_id, nodes in self.communities.items()
            if len(nodes) < min_size
        ]
        
        for comm_id in small_communities:
            nodes = self.communities[comm_id]
            neighbor_communities = defaultdict(int)
            for node in nodes:
                for neighbor in graph.neighbors(node):
                    neighbor_comm = self.node_to_community.get(neighbor)
                    if neighbor_comm is not None and neighbor_comm != comm_id:
                        neighbor_communities[neighbor_comm] += 1
            
            if neighbor_communities:
                best_neighbor = max(neighbor_communities.items(), key=lambda x: x[1])[0]
                self.communities[best_neighbor].update(nodes)
                del self.communities[comm_id]
        
        self._build_node_to_community_mapping()
        logger.info(f"After merging: {len(self.communities)} communities")

