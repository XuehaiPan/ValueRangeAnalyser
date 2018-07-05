import pygraphviz as pgv


# strict (no parallel edges)
# digraph
# with attribute rankdir set to 'LR'
G = pgv.AGraph(directed = True, strict = True)
G.add_nodes_from(['1', '2', '3', '4', '5'], shape = 'box')
G.add_node('entry', shape = 'ellipse')
G.add_node('exit', shape = 'ellipse')

G.add_edge('entry', '1')
G.add_edge('5', 'exit')
G.add_edge('1', '2')
G.add_edge('2', '3')
G.add_edge('2', '4')
G.add_edge('3', '4')
G.add_edge('4', '5')

G.layout(prog = 'dot')  # layout with dot
G.draw(path = 'pygraphvizDemo.png')  # write to file
