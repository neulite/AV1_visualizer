def combine_name(row, df_node_type):
    node_type_id = str(int(row["node_type"]))
    #name = df_node_type[df_node_type["node_type_id"]==node_type_id]
    name = df_node_type.query(f"node_type_id == {node_type_id}").iloc[0]["pop_name"]
    #print(name)
    
    return name+ "_" + node_type_id

def split_name(s):
    return s.split("_")[0]

def int_to_list(n):
    return range(n)


