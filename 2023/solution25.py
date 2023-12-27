import networkx as nx


def read_input():
    with open("input25.txt") as f:
        puzzle_input = f.read()

    return puzzle_input


def parse(s):
    G = nx.Graph()
    for line in s.split("\n"):
        u, mess = line.split(": ")
        for v in mess.split():
            G.add_edge(u, v)
    return G


def snip(G, n=3):
    G = G.copy()
    edge_centralities = sorted(nx.edge_betweenness_centrality(G).items(), key=lambda t: -t[1])

    for (u, v), _ in edge_centralities[:n]:
        G.remove_edge(u, v)

    components = nx.connected_components(G)
    return components


def multiply_component_sizes(components):
    res = 1
    for component in components:
        res *= len(component)
    return res


def main():
    raw = read_input()
    G = parse(raw)

    parts = snip(G)
    star1 = multiply_component_sizes(parts)
    print(f"The product of the number of nodes in the component is: {star1}.")


if __name__ == '__main__':
    main()
