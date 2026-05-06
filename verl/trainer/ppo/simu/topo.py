from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from verl.trainer.ppo.simu.network_op import LogicalTransfer


@dataclass
class HostTopo:
    intra_host_bandwidth: float # bandwidth of intra-host communication in Gbps
    intra_host_latency: float # latency of intra-host communication in ms
    hosts_connections: dict[tuple[int, int], Path] # directed mapping from (host_id_1, host_id_2) to the path between them
        # Assuming deterministic routing

    def get_simultaneous_system_flow_time(self, transfers: list[LogicalTransfer]) -> dict[tuple[int, int], float]:
        """
        Gives the times to complete each connection assuming they all start simultaneously and share bandwidth fairly when paths overlap.
        Used for modeling completion time for a series of inter-stage communications.

        transfers: list of LogicalTransfer objects
        """ 
        # Get loads on each link
        link_loads: defaultdict[Link, float] = defaultdict(float)
        intra_host_loads: defaultdict[int, float] = defaultdict(float)
        for transfer in transfers:
            host_a = transfer.src_host_id
            host_b = transfer.dst_host_id
            data_size = transfer.data_size_GB
            if host_a == host_b:
                intra_host_loads[host_a] += data_size
            else:
                path = self.hosts_connections[(host_a, host_b)]
                for link in path.path:
                    link_loads[link] += data_size

        # Get time for each link to clear
        link_times: dict[Link, float] = {}
        for link, load in link_loads.items():
            # Theory: total time = (total data in GB * 8) / bandwidth in Gbps
            link_times[link] = (load * 8) / link.bandwidth
            
        intra_times: dict[int, float] = {}
        for host, load in intra_host_loads.items():
            intra_times[host] = (load * 8) / self.intra_host_bandwidth
        
        # Get time for each connection
        connection_times: dict[tuple[int, int], float] = {}
        for transfer in transfers:
            host_a = transfer.src_host_id
            host_b = transfer.dst_host_id
            data_size = transfer.data_size_GB
            if host_a == host_b:
                time = intra_times[host_a]
                time += self.intra_host_latency / 1000.0
                connection_times[(host_a, host_b)] = time
            else:
                path = self.hosts_connections[(host_a, host_b)]
                time = max(link_times[link] for link in path.path) # connection time is dominated by slowest link
                # Add path latency (convert ms to seconds)
                time += path.latency / 1000.0
                connection_times[(host_a, host_b)] = time

        return connection_times
    
    def get_simultaneous_system_shared_bandwidth(self, transfers: list[LogicalTransfer]) -> dict[LogicalTransfer, float]:
        """
        Gives the effective bandwidth for each connection assuming they all start simultaneously and share bandwidth fairly when paths overlap.
        Used for modeling communication overhead intra-stage under continuous communication.  Most accurate when transfers are of similar magnitude,
        approaches worst-case estimate otherwise.

        transfers: list of LogicalTransfer objects
        """ 
        # Get loads on each link
        link_loads: defaultdict[Link, int] = defaultdict(int)
        intra_host_loads: defaultdict[int, int] = defaultdict(int)
        for transfer in transfers:
            host_a = transfer.src_host_id
            host_b = transfer.dst_host_id
            if host_a == host_b:
                intra_host_loads[host_a] += 1
            else:
                path = self.hosts_connections[(host_a, host_b)]
                for link in path.path:
                    link_loads[link] += 1

        # Get effective bandwidth for each connection
        connection_bandwidths: dict[LogicalTransfer, float] = {}
        for transfer in transfers:
            host_a = transfer.src_host_id
            host_b = transfer.dst_host_id
            if host_a == host_b:
                connection_bandwidths[transfer] = self.intra_host_bandwidth # TODO: Enhance, currently assume nodes are limited to one-intra-host transfer at a time
            else:
                path = self.hosts_connections[(host_a, host_b)]
                bandwidths = []
                for link in path.path:
                    # Theory: effective bandwidth = total bandwidth / number of connections sharing the link
                    effective_bandwidth = link.bandwidth / link_loads[link]
                    bandwidths.append(effective_bandwidth)
                connection_bandwidths[transfer] = min(bandwidths) # connection bandwidth is dominated by slowest link

        return connection_bandwidths

    def get_simultaneous_logical_transfer_times(self, transfers: list[LogicalTransfer]) -> dict[LogicalTransfer, float]:
        """
        Gives the completion times for a list of logical transfers executed simultaneously, 
        evaluating both bandwidth bottlenecks and latency constraints.
        """
        connection_bandwidths = self.get_simultaneous_system_shared_bandwidth(transfers)
        
        transfer_times: dict[LogicalTransfer, float] = {}
        for t in transfers:
            bw = connection_bandwidths[t]
            if t.src_host_id == t.dst_host_id:
                latency_s = self.intra_host_latency / 1000.0
            else:
                latency_s = self.hosts_connections[(t.src_host_id, t.dst_host_id)].latency / 1000.0
            transfer_times[t] = (t.data_GB * 8) / bw + latency_s
            
        return transfer_times

@dataclass
class Path:
    path: list[Link] # list of nodes in the path
    bandwidth: float # bandwidth of the path in Gbps
    latency: float # latency of the path in ms

    def __init__(self, link: list[Link]):
        self.path = link
        for i in range(len(link) - 1):
            if link[i].node_b_id != link[i + 1].node_a_id:
                raise ValueError(f"Invalid path: {link[i].node_b_id} != {link[i + 1].node_a_id}")
            
        self.bandwidth = min(link[i].bandwidth for i in range(len(link)))
        self.latency = sum(link[i].latency for i in range(len(link)))

@dataclass
class Link:
    node_a_id: int # node A ID
    node_b_id: int # node B ID
    bandwidth: float # bandwidth of the link in Gbps
    latency: float # latency of the link in ms