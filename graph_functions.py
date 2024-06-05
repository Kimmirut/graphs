from sys import stdin, stdout
from collections import deque


# DFS algorythms.


# 1. DFS algorythms that work on adjacency matrix based graphs

def dfs_on_matrix(vertex: int, matrix: list) -> list:
    '''Returns dfs result for given matrix from given vertex.'''

    passed: list = [False for _ in range(len(matrix))]
    vertices: list = [vertex]
    passed[vertex] = True

    def dfs(vertex) -> None:
        '''Fucntion, that fills passed list.'''

        for v, is_passed in enumerate(matrix[vertex]):
            if is_passed and not passed[v]:
                passed[v] = True
                vertices.append(v)
                dfs(v)
    dfs(vertex)

    return vertices


def dfs_on_matrix_it(graph: list) -> list:
    '''Iterative DFS on matrix'''

    vert_n: int = len(graph)
    used: list = [False for _ in range(vert_n)]
    passed_vertices = []

    stack = []
    # Sort through vertices
    for vertex in range(vert_n):
        if used[vertex]:                   # if it's already used,
            continue                       # omits it
        stack.append(vertex)               #

        while len(stack) != 0:  # dfs started
            v = stack.pop()
            if used[v]:
                continue
            used[v] = True                # mark it as used
            passed_vertices.append(v)
            for adj_vertex, is_connected in enumerate(graph[v]): #   run dfs for all v neighbours
                if is_connected and not used[adj_vertex]:
                    stack.append(adj_vertex)

    return passed_vertices


def get_conn_comps(graph: list) -> list:
    '''Returns all connectivity components for adjacity list based for given UNORIENTED graph.'''


    def dfs_on_adj_list(vertex: int, adj_list: list) -> list:
        '''DFS algorythm that returns dfs from given vertex.'''

        passed_vertices: list = [vertex]
        used: list = [False for _ in range(len(adj_list))]
        used[vertex] = True

        def dfs(vertex) -> None:
            for v in adj_list[vertex]:
                if not used[v]:
                    used[v] = True
                    passed_vertices.append(v)
                    dfs(v)

        dfs(vertex)
        return tuple(passed_vertices)


    used: list = [False for _ in range(len(graph))]
    conn_comps: list[tuple] = list()

    for vertex in range(len(graph)):
        if graph[vertex] == []:
            used[vertex] = True
            conn_comps.append((vertex,))
            continue
        if not used[vertex]:
            conn_comp: tuple[int] = dfs_on_adj_list(vertex, graph)
            for v in conn_comp:
                used[v] = True
            conn_comps.append(conn_comp)

    return conn_comps


def has_cycle(adj_matrix: list[list]) -> bool:
    '''Function that takes adjacity matrix based graph,
    and returns True if there's a cycle, else False, numeration from 0'''

    def vertex_has_cycle(vertex: int) -> None:
        '''Returns True if given vertex has cycle else False'''
        if colors[vertex] == BLACK:
            return

        stack = [vertex]
        colors[vertex] = GRAY
        while len(stack) != 0:
            curr_v = stack.pop()
            for v, is_linked in enumerate(adj_matrix[curr_v]):
                if is_linked:
                    if colors[v] == BLACK:
                        continue
                    if colors[v] == GRAY:
                        return True
                    stack.append(v)
        colors[vertex] = BLACK
        return False


    WHITE, GRAY, BLACK = 0, 1, 2
    colors = [WHITE for _ in range(len(adj_matrix))]

    for vertex in range(len(adj_matrix)):
        if vertex_has_cycle(vertex) == True:
            return True

    return False


# ----------------------------------------------------------------------------------------------------------
# 2. DFS algorythms that work on adjacency list based graphs


def dfs_on_adj_list(vertex: int, adj_list: list) -> list:
    '''Returns True if given vertex is cycled, else False'''

    if colors[vertex] == GRAY:
        return True
    if colors[vertex] == BLACK:
        return False

    colors[vertex] = GRAY
    for v, is_linked in enumerate(adj_matrix[vertex]):
        if is_linked and colors[v] != BLACK:
            if colors[v] == GRAY:
                return True
            vertex_cycled(v)

    colors[vertex] = BLACK

    WHITE, GRAY, BLACK = 0, 1, 2
    colors: list = [WHITE for _ in range(len(adj_matrix))]
    for vertex in range(len(adj_matrix)):
        if vertex_cycled(vertex):
            return True

    return False


def get_cycle(adj_matrix: list[list]) -> list|None:
    '''Returns cycle if given adjacity matrix based graph is cycled, else None.'''

    def vertex_cycle(vertex: int, path: list) -> list|None:
        '''Returns cycle if given vertex is cycled, else None'''

        if colors[vertex] == BLACK:
            return None

        colors[vertex] = GRAY
        for v, is_linked in enumerate(adj_matrix[vertex]):
            if is_linked and colors[v] != BLACK:
                if colors[v] == GRAY:
                    return path[path.index(v):]
                res = vertex_cycle(v, path + [v])
                if res is not None:
                    return res

        colors[vertex] = BLACK
        return False

    WHITE, GRAY, BLACK = 0, 1, 2
    colors: list = [WHITE for _ in range(len(adj_matrix))]
    for vertex in range(len(adj_matrix)):
        res = vertex_cycle(vertex, [vertex])
        if res is not None:
            return res


def topological_sort(adj_matrix: list[list]) -> list:
    '''Returns topological order for given adjacity matrix based graph.'''

    def dfs(vertex: int, adj_matrix: list[list], topological_order: list) -> None:
        '''fills the topological order of given adj matrix based graph'''

        used[vertex] = True
        for v, is_linked in enumerate(adj_matrix[vertex]):
            if is_linked and not used[v]:
                dfs(v, adj_matrix, topological_order)
        topological_order.append(vertex)

    used = [False for _ in range(len(adj_matrix))]
    topological_order = []
    for vertex in range(len(adj_matrix)):
        if not used[vertex]:
            dfs(vertex, adj_matrix, topological_order)

    return topological_order


def topological_sort(adj_list: dict[int, list]) -> list:
    '''Returns topological order for given adjacity list based graph.'''

    def dfs(vertex: int, adj_list: dict[int, list], topological_order: list) -> None:
        '''fills the topological order of given adj list based graph'''

        if not used[vertex]:
            used[vertex] = True
            for v in adj_list[vertex]:
                if not used[v]:
                    dfs(v, adj_list, topological_order)

            topological_order.append(vertex)


    used = [False for _ in range(len(adj_list))]
    topological_order = []
    for vertex in range(len(adj_list)):
        dfs(vertex, adj_list, topological_order)

    return topological_order


def is_cycled(adj_list: list[list]) -> bool:
    '''Returns True if given adjacity list based graph is cycled, else False.'''

    def vertex_cycled(vertex) -> bool:
        '''DFS based algorythm that returns True if any cycle starts from given vertex, else False.'''

        if colors[vertex] == BLACK:
            return

        colors[vertex] = GRAY
        for v in adj_list[vertex]:
            if colors[v] == BLACK:
                continue
            if colors[v] == GRAY:
                return True
            if vertex_cycled(v):
                return True
            colors[vertex] = BLACK


    WHITE, GRAY, BLACK = 0, 1, 2
    colors = [WHITE for _ in range(len(adj_list))]
    for vertex in range(len(adj_list)):
        if not colors[vertex] == BLACK and vertex_cycled(vertex):
            return True

    return False


def acyclic_dijkstra(vertex: int, adj_matrix: list[list]) -> list:
    '''
    Returns an array that contains shortest paths from given
    vertex to every other. Absence of path between vertices should
    be denoted as 0.
    '''

    def bfs(vertex: int) -> None:
        '''BFS algorythm that fills "dist" list.'''

        queue: deque[int] = deque([vertex])
        while len(queue) != 0:
            curr_v = queue.popleft()
            used[vertex] = True
            for v, weight in enumerate(adj_matrix[curr_v]):
                if weight == 0 or used[v] is True :  # path doesn't exist or vertex was used
                    continue
                dist[v] = min(dist[curr_v] + weight, dist[v])
                queue.append(v)


    dist = [0] + [float('inf') for _ in range(len(adj_matrix) - 1)]
    used = [False for _ in range(len(adj_matrix))]
    bfs(vertex)

    return dist


def dijkstra(vertex: int, adj_matrix: list[list]) -> list:
    '''
    Returns an array that contains shortest paths from given
    vertex to every other. Absence of path between vertices should
    be denoted as 0.
    '''

    def get_min_unused_indx(array: list, used: list) -> int:
        mn, mn_index = float('inf'), -1
        for i, item in enumerate(array):
            if used[i]:
                continue
            if item < mn:
                mn = item
                mn_index = i

        return mn_index

    dist = [float('inf') for _ in range(len(adj_matrix))]
    used = [False for _ in range(len(adj_matrix))]
    dist[vertex] = 0
    used[vertex] = True

    while vertex != -1:
        for v, weight in enumerate(adj_matrix[vertex]):
            if weight == 0 or used[v] is True :  # path doesn't exist or vertex was used
                continue
            dist[v] = min(dist[vertex] + weight, dist[v])
        used[vertex] = True
        vertex = get_min_unused_indx(dist, used)

    return dist


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# Graph converters.

def matrix_to_edges_list(matrix: list) -> list:
    '''Converts given adjacency matrx in edges list, and returns it.'''

    edges_list: list = []
    n: int = int(stdin.readline())

    for vertex in range(1, n + 1):
        row: list = list(map(int, stdin.readline().split()))
        for i, v_i in enumerate(row, start=1):
            if v_i == 1:
                edges_list.append((vertex, i))

    return edges_list


def edge_list_to_adj_list(edge_list):
        adj_list = {}
        for edge in edge_list:
            start, end = edge
            adj_list[start] = adj_list.get(start, []) + [end]
            adj_list[end] = adj_list.get(end, []) + [start]


def edge_list_to_adj_matrix() -> list[list]:
    v, e = map(int, input().split())
    adj_matrix = [[0 for _ in range(v)] for _ in range(v)]
    for _ in range(e):
        start_v, end_v = map(int, input().split())
        adj_matrix[start_v][end_v] = 1
    return adj_matrix


# Other graph algorytms.

def Floyd_algorythm(G: list[list[int]], on_place: bool=True) -> list[list[int]]:
    '''
    Returns shortest path lenght from all vertex to all other vertices of graph.
    Works on adjacity matrix based weighted garph without negative cycles.
    '''

    if not on_place:
        from copy import deepcopy

        G = deepcopy(G)

    v = len(G)
    for k in range(v):
        for i in range(v):
            for j in range(v):
                G[i][j] = min(G[i][j], G[i][k] + G[k][j])

    if not on_place:
        return G


# Graph examples for testing.

# adj_list = [
#     [1],
#     [0, 2, 3],
#     [1],
#     [1]
# ]

# adj_matrix = [
#     [0, 1, 0, 0],
#     [1, 0, 1, 1],
#     [0, 1, 0, 0],
#     [0, 1, 0, 0],
# ]
# adj_list = [
#     [2],
#     [2],
#     [0, 1, 3],
#     [2],
# ]

# adj_matrix = [
#     [0, 1, 0, 0, 0],
#     [1, 0, 1, 1, 0],
#     [0, 1, 0, 0, 0],
#     [0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ]

# adj_matrix = [
#     [0, 1, 1],
#     [1, 0, 1],
#     [1, 1, 0]
# ]

# cycled_graph = [
#     [0, 1, 1],
#     [0, 0, 1],
#     [1, 0, 0]
# ]

# cycled_graph = [
#     [0, 1, 1, 0, 0],
#     [0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 0]
# ]


# print(has_cycle(cycled_graph))

# a = [[0, 1, 0, 0, 1],
#      [0, 0, 1, 0, 0],
#      [0, 0, 0, 1, 0],
#      [0, 0, 0, 0, 1],
#      [0, 1, 0, 0, 0],]

# print(has_cycle(a))
