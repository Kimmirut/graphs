class Graph:

    '''Adjacency matrix based graph.'''

    def __init__(self, v) -> None:
        self.vertex_number = v
        self.__G = [[0 for _ in range(v)] for _ in range(v)]

    def is_unoriented(self):
        n = self.vertex_number
        for i in range(1, n):
            for j in range(i):
                if self.__G[i][j] != self.__G[j][i]:
                    return False
        return True

    def print_graph(self):
        item_lenght = self.__get_max_lenght(self.__G) + 1
        n = self.vertex_number
        print('  ' + ' '.join([str(i).rjust(item_lenght, ' ') for i in range(1, n + 1)]))
        for i, row in enumerate(self.__G, start=1):
            print(i, end=' ')
            for item in row:
                print(str(item).rjust(item_lenght, ' '), end=' ')
            print()

    def __getitem__(self, edge):
        i, j = self.__validate_indices(edge)

        return self.__G[i-1][j-1]

    def __setitem__(self, edge, value):
        if type(value) != int:
            raise ValueError(f'edge argument should an integer 0 or 1, {value} of type {type(value)} was given')

        i, j = self.__validate_indices(edge)
        self.__G[i-1][j-1] = value

    def __delitem__(self, edge):
        self.__setitem__(edge, 0)

    def __validate_indices(self, edge):
        try:
            i, j = edge
        except (TypeError, ValueError) as e:
            raise type(e)('Graph\'s edge should look like: graph[v1, v2]')

        if type(i) != int or type(j) != int:
            raise TypeError('Vertexes should be an integers')

        if not (0 < i <= self.vertex_number) or not (0 < j <= self.vertex_number):
            raise ValueError('No such vertex in graph')

        return i, j

    @staticmethod
    def __get_max_lenght(g):
        mx = 1
        for row in g:
            row_max = len(str(max(row, key=lambda row: len(str(row)))))
            if mx < row_max:
                mx = row_max
        return mx


class Graph:

    '''
    List of edges based graph class.\n
    Supports all basic list operations.\n
    Oreintation and weighting must be indicated while creating.
    '''

    def __init__(self, edges_list: list = None, oriented: bool = False, weighted: bool = False) -> None:
        '''
        In case of unweighted graph, edges list should look like:
        [(start1, end1), (start2, end2), ...,  (start_n, end_n)]\n
        if weighted:
        [(start1, end1, weight1), (start2, end2, weight2), ...,  (start_n, end_n, weight_n)]
        '''

        self.__validate_init_args(oriented, weighted, edges_list)
        self.orinted = oriented
        self.weighted = weighted
        self._lst = []
        if edges_list is not None:
            for edge in edges_list:
                self.append(edge)

    # Default list methods with edge validation.

    def append(self, edge: tuple):
        '''Appending edge to list if it wasn't there.'''

        self.__validate_edge(edge, self.weighted)
        for item in self:
            if item == edge:
                break
        else:
            self._lst.append(edge)

    def remove(self, edge: tuple):
        '''Removing given edge from a list, throws an excpetion if it wasn't in list'''

        self.__validate_edge(edge, self.weighted)
        for i, item in enumerate(self):
            if item == edge:
                self._lst.pop(i)
                break
        else:
            raise ValueError(f'Edge - {edge} is not in list')

    def __repr__(self) -> str:
        return f'Graph(edges_list={self._lst}, oriented = {self.orinted}, weighted = {self.weighted})'

    def __eq__(self, __value: object) -> bool:
        if not isinstance(self, Graph) or not isinstance(__value, Graph):
            raise TypeError('Can only compare graph with graph')

        if self.weighted != __value.weighted:
            return False

        return self._lst == __value._lst

    def __contains__(self, edge: tuple) -> bool:
        self.__validate_edge(edge, self.weighted)
        return edge in self._lst

    # Default list methods that just delegate their calls to list.

    def __iter__(self):
        return iter(self._lst)

    def __getitem__(self, item):
        return self._lst[item]

    def __setitem__(self, key, value):
        self._lst[key] = value

    def __delitem__(self, item):
        del self._lst[item]

    # Validation methods.

    def __validate_init_args(self, oriented, weighted, edges_list):
        if not isinstance(oriented, bool):
            raise TypeError(f'"oriented" argument should be boolean, not {oriented}, of {type(oriented)} type')

        if not isinstance(weighted, bool):
            raise TypeError(f'"weighted" argument should be boolean, not {weighted}, of {type(weighted)} type')

        if edges_list is not None:
            if not isinstance(edges_list, list):
                raise TypeError('Edges list is not a list')

            for edge in edges_list:
                self.__validate_edge(edge, weighted)


    def __validate_edge(self, edge: tuple, weighted: tuple):
        start, end = edge[:2]
        if len(edge) == 3 and not weighted:
            raise ValueError('Weight was given in a list of edges, but graph is unweighted')

        if len(edge) == 2 and weighted:
            raise ValueError('Weight was not given in a list of edges, but graph is weighted')

        if weighted:
            weight = edge[2]
            if type(weight) not in (int, float):
                raise TypeError(f'{(start, end, weight)} weight should be a number')

        if type(start) != int:
                raise TypeError(f'{(start, end, weight)} start should be a number')

        if type(end) != int:
                raise TypeError(f'{(start, end, weight)} end should be a number')


class AdjList(list):

    '''Adjacity list based graph.'''

    def __init__(self, oriented: bool=False, weighted: bool=False,
                  adj_list: list[int|None] = None) -> None:
        '''adj_list should be valid adjacity list, containing list of ints.
        Validation is based on given "weighted" and "oriented" arguments'''

        if not isinstance(oriented, bool):
            raise TypeError(f'"oriented" argument shuld be a boolean, {oriented} was given')

        if not isinstance(weighted, bool):
            raise TypeError(f'"oriented" argument shuld be a boolean, {oriented} was given')

        self.oriented = oriented
        self.weighted = weighted
        if adj_list is not None:
            self.__validate_adg_list(adj_list)
            self.adj_list = adj_list

    def __validate_adg_list(self, adj_list: list[int|None]):
        try:
            iter(adj_list)
        except TypeError:
            raise TypeError('given "adj_list" argument isn\'t iterable')

        for vertex, adg_vertices in enumerate(adj_list):    # adg_vertices should look like: [v1, v2, ...]
            for adj_vertex in adg_vertices:  # vartex should look like: v1 if unweighted, else: (v1, weight)
                if self.weighted:
                    if not isinstance(adj_vertex, tuple):
                        raise TypeError(f'vertex: {adj_vertex}, type of {type(adj_vertex)} in given adj_list isn\'t tuple')

                    if len(adj_vertex) != 2:
                        raise ValueError('unweighted graph should contain vertices like: (start: int, weight: int|float)')

                    self.__validate_vertex(adj_vertex[0])

                    weight = adj_vertex[1]
                    if type(weight) not in (int, float):
                        raise TypeError(f'weight in vertex should be a number, {weight}, type of {type(weight)} was given')

                else:
                    self.__validate_vertex(adj_vertex)

                # Further code checking if graph is unoriented if given "oriented" argument is True.

                if not self.oriented:
                    if self.weighted:
                        if (vertex, weight) not in adj_list[adj_vertex[0]]:
                            raise ValueError('Graph is oriented when "oriented" argument is False')
                    else:
                        if vertex not in adj_list[adj_vertex]:
                            raise ValueError('Graph is oriented when "oriented" argument is False')

    def __validate_vertex(self, vertex: int) -> None:
        '''Checks if vertex is int and if its in range of graph vertexes, raises an error if not.'''

        if type(vertex) != int:
            raise TypeError(f'Each elemnt of adg_list should contain only ints, {vertex} was given')

        if not (0 <= vertex < len(self.adj_list)):
            raise ValueError(f'unappropriate vertex argument: {vertex}')

    def get_conn_comps(self) -> list:
        '''Returns all connection components for adjacity list based for given UNORIENTED graph.'''


        def dfs_on_adj_list(vertex: int, adj_list: list) -> list:
            '''DFS algorythm that returns dfs from given vertex.'''

            passed_vertices: set = [vertex]
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

        if self.oriented:
            raise NotImplementedError
        graph = self.adj_list
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
