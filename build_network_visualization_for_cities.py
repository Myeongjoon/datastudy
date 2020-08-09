import os
import random
import graph_tool.all as gt


def process_one_city(city_name, which=1, edges_percent=0.01, layout=1):
    g = gt.Graph()

    # load userid_bizid_star_tuple_for_graph
    base_dir = "/Users/sundeepblue/Bootcamp/allweek/week9/capstone/data/yelp_data/split_business_data_by_city/"

    folder_name = "userid_businessid_star_tuple.csv" # note this is actually a folder not csv file
    full_folder_path = os.path.join(base_dir, city_name, folder_name)
    files = os.listdir(full_folder_path)
    tuples = []
    for f in files:
        if f.startswith("part-"):
            full_path = os.path.join(full_folder_path, f)
            print full_path
            with open(full_path, "r") as h:
                lines = h.readlines()
            tuples += lines
    print len(tuples)

    # populate edge list by parsing tuples
    edge_list = []
    for t in tuples:
        sp = t.strip().split(",")
        user_id = int(sp[0].strip())
        business_id = int(sp[1].strip())
        edge_list.append((user_id, business_id))

    num_edges = len(edge_list)
    num_chosen_edges = int(num_edges * edges_percent)

    # decide how many edges and in what type to be included in network
    if which == 1:
        random_indices = random.sample(range(len(edge_list)), num_chosen_edges)
        partial_edge_list = [edge_list[i] for i in random_indices]
        graph_type = "random_{}_edges".format(num_chosen_edges)
        print "{} edges were randomly chosen and added into graph!".format(num_chosen_edges)
    elif which == 2:
        partial_edge_list = edge_list[:num_chosen_edges]
        print "first {} edges were chosen and added into graph!".format(num_chosen_edges)
        graph_type = "first_{}_edges".format(num_chosen_edges)
    else: # caution. this option is very time consuming if nodes number are big!
        partial_edge_list = edge_list
        print "use all {} edges to build graph!".format(len(edge_list))
        graph_type = "all_{}_edges".format(len(edge_list))


    g.add_edge_list(partial_edge_list)

    # Remove invalid vertices that added by graph-tool itself
    # Eg, if we add an edge by running "e = g.add_edge(1000, 2000)", by default, graph-tool will automatically
    # add 2000 vertices indexed from 1 to 2000. However, we only need two vertices (of indix 1000 and index 2000).
    # The code below scan all vertices, and mark those with non-zero indegree/outdegree as valid ones.

    print "Removing invalid vertices ..."
    valid_nodes = filter(lambda v: v.in_degree() != 0 or v.out_degree() != 0, g.vertices())
    keep = g.new_vertex_property('bool')
    for v in valid_nodes:
        keep[v] = True
    g.set_vertex_filter(prop=keep)

    print g.num_vertices()
    print g.num_edges()

    # do not show vertex text
    print "Exporting graph image ..."
    output_file_name = "{}_graph_{}_{}.pdf".format(city_name, graph_type, layout)
    if layout == 1:
        pos = gt.sfdp_layout(g)
    else:
        pos = gt.arf_layout(g)

    # output network graph
    gt.graph_draw(g, pos=pos, output_size=(2000, 2000), 
        vertex_color=[1,1,1,0], vertex_size=2, edge_pen_width=1, output=output_file_name)
    print "Done!"


# process all major cities
us_cities = [
                ("NC", "us_charlotte"), 
                ("NV", "us_lasvegas"), 
                ("WI", "us_madison"),
                ("AZ", "us_phoenix"), 
                ("PA", "us_pittsburgh"), 
                ("IL", "us_urbana_champaign")
            ]
canada_cities = [("QC", "canada_montreal")]
germany_cities = [("BW", "germany_karlsruhe")]
uk_cities = [("EDH", "uk_edinburgh")]

cities = us_cities + canada_cities + germany_cities + uk_cities
city_names = [p[1] for p in cities]

which = 1
layout = 2
edges_percent = 0.01

for city_name in city_names:
    process_one_city(city_name, which, edges_percent, layout)
print "Done!"

