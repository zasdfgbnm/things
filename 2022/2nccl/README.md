# Step 1: Setup network environment

Create two network namespaces, emulationg two nodes:

```bash
ip netns add test1
ip netns add test2
```

Create two pairs of virtual interfaces

```bash
ip link add test1net1 type veth peer name test2net1
ip link add test1net2 type veth peer name test2net2
```

Move interfaces to namespaces

```
ip link set test1net1 netns test1
ip link set test1net2 netns test1
ip link set test2net1 netns test2
ip link set test2net2 netns test2
```

Setup IP addresses for all interfaces

```bash
ip netns exec test1 ip addr add 192.168.101.1/24 dev test1net1
ip netns exec test1 ip addr add 192.168.102.1/24 dev test1net2
ip netns exec test2 ip addr add 192.168.101.2/24 dev test2net1
ip netns exec test2 ip addr add 192.168.102.2/24 dev test2net2
```

Bring up all interfaces

```bash
ip netns exec test1 ip link set dev test1net1 up
ip netns exec test1 ip link set dev test1net2 up
ip netns exec test2 ip link set dev test2net1 up
ip netns exec test2 ip link set dev test2net2 up
```

Use ping to validate network connectivity

In one command line window:

```bash
$ ip netns exec test1 tcpdump -i test1net1
tcpdump: verbose output suppressed, use -v[v]... for full protocol decode6
18:23:14.661862 ARP, Request who-has 192.168.101.1 tell 192.168.101.2, length 28
18:23:14.661883 ARP, Reply 192.168.101.1 is-at f2:fa:4a:06:fe:c3 (oui Unknown), length 28
18:23:14.661886 IP 192.168.101.2 > 192.168.101.1: ICMP echo request, id 49895, seq 1, length 64
18:23:14.661896 IP 192.168.101.1 > 192.168.101.2: ICMP echo reply, id 49895, seq 1, length 64
18:23:15.687023 IP 192.168.101.2 > 192.168.101.1: ICMP echo request, id 49895, seq 2, length 64
18:23:15.687046 IP 192.168.101.1 > 192.168.101.2: ICMP echo reply, id 49895, seq 2, length 64
```

In another command line window:

```bash
$ ip netns exec test2 ping 192.168.101.1
PING 192.168.101.1 (192.168.101.1) 56(84) bytes of data.
64 bytes from 192.168.101.1: icmp_seq=1 ttl=64 time=0.056 ms
64 bytes from 192.168.101.1: icmp_seq=2 ttl=64 time=0.073 ms
```

In one command line window:

```bash
$ ip netns exec test1 tcpdump -i test1net2
tcpdump: verbose output suppressed, use -v[v]... for full protocol decode
listening on test1net2, link-type EN10MB (Ethernet), snapshot length 262144 bytes
18:25:00.811566 ARP, Request who-has 192.168.102.1 tell 192.168.102.2, length 28
18:25:00.811583 ARP, Reply 192.168.102.1 is-at 86:8b:f9:56:14:1e (oui Unknown), length 28
18:25:00.811585 IP 192.168.102.2 > 192.168.102.1: ICMP echo request, id 669, seq 1, length 64
18:25:00.811593 IP 192.168.102.1 > 192.168.102.2: ICMP echo reply, id 669, seq 1, length 64
```

In another command line window:

```bash
$ ip netns exec test2 ping 192.168.102.1
PING 192.168.102.1 (192.168.102.1) 56(84) bytes of data.
64 bytes from 192.168.102.1: icmp_seq=1 ttl=64 time=0.044 ms
```

# Step 2: Run testing scripts

In one command window:

```bash
ip netns exec test1 sudo -u gaoxiang $PWD/start1.sh
```

In another command window:

```bash
ip netns exec test2 sudo -u gaoxiang $PWD/start2.sh
```