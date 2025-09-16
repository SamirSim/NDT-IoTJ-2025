# This is the final version of the script to generate the number of transmissions series per configuration for each node.
import json
import time

for i in ["4"]:
    file_title = 'expe-iotj-1ps-'+i
    filename = '../data/IoTJ/' + file_title + '.txt'
    output_json_file = '../data/IoTJ/' + file_title + '-transmissions-per-config-series.json'
    output_json_file_stats = '../data/IoTJ/' + file_title + '-transmissions-per-config-stats.json'

    # Read and parse logs
    with open(filename, 'r') as file:
        data = file.readlines()

    node_mac_map = {
        "m3-97": "b179",
        "m3-98": "b069",
        "m3-99": "b277",
        "m3-100": "b080",
        "m3-101": "9181",
        "m3-109": "1062",
        "m3-110": "a071",
        "m3-111": "9172",
        "m3-112": "a578",
        "m3-119": "a468",
        "m3-120": "a469",
        "m3-121": "8572",
        "m3-122": "a180",
        "m3-129": "9783",
        "m3-130": "a272",
        "m3-131": "9369",
        "m3-133": "2360",
        "m3-134": "a069",
        "m3-143": "9779",
        "m3-153": "b081",
        "m3-154": "a279",
        "m3-156": "1861",
        "m3-157": "a380",
        "m3-167": "1862",
        "m3-168": "8475",
        "m3-169": "a276",
        "m3-170": "c076",
        "m3-171": "9379"
    }

    # Initialize variables
    node_data = {}
    current_config = {}

    for line in data:
        config = None
        parts = line.split(";")
        if len(parts) < 3:
            continue

        timestamp, node_id, message = parts

        if "The configuration has been changed to" in message:
            config = tuple(map(int, [
                message.split("CSMA_MIN_BE=")[1].split(",")[0],
                message.split("CSMA_MAX_BE=")[1].split(",")[0],
                message.split("CSMA_MAX_BACKOFF=")[1].split(",")[0],
                message.split("FRAME_RETRIES=")[1].split()[0]
            ]))
            config = str(config)
            current_config[node_id] = config
            if node_id not in node_data:
                node_data[node_id] = {}
            if config not in node_data[node_id]:
                node_data[node_id][config] = {}

        elif "Sending packet content:" in message:
            packet_id = message.split(": ")[1].strip("\n").strip("'")
            config = current_config.get(node_id, None)
            try:
                #if packet_id == "0103324" and node_id == "m3-143":
                #    print(timestamp, node_id, config, packet_id)
                    
                node_data[node_id][config][packet_id] = {}
                if config:
                    node_data[node_id][config][packet_id]["received"] = 0
                    node_data[node_id][config][packet_id]["transmissions"] = 0
            except:
                #print("Error for: ", timestamp, node_id, config, packet_id)
                pass
        
        # NB: Fixed here: Put the configuration of a reception as the one in which the packet was received and not sent
        elif "Data received from" in message:
            packet_id = message.split(" '")[1].strip("\n").strip("'")
            try:
                mac = message.split("from fd00::")[1].split()[0]
                node = next((n for n, m in node_mac_map.items() if m == mac), None)
                #print(node, packet_id, mac)
                if node:
                    #for config in node_data.get(node, {}):
                    #    if packet_id in node_data[node][config]: # Added this here Only consider packets sent with the current configuration
                    #        node_data[node][config][packet_id]["received"] = 1
                    node_data[node][current_config[node]][packet_id]["received"] = 1

            except:
                #print("Error for: ", timestamp, node_id, packet_id)
                pass # Ignore messages coming from other nodes such as 9181 (m3-101)

        elif "csma ok" in message:
            packet_id = message.split(": ")[2].strip("\n").strip("'")
            x = int(message.split("ok: ")[1].split(" for")[0])

            try:
                node_data[node_id][current_config[node_id]][packet_id]["transmissions"] = 1/x
            except:
                #print("Error for: ", timestamp, node_id, packet_id)
                pass
            
            #for config in node_data.get(node_id, {}):
            #    if packet_id in node_data[node_id][config]:
            #        node_data[node_id][config][packet_id]["transmissions"] = 1/x

    # Keep only the transmissions data for the received packets in a new dict
    new_node_data = {}
    for node, config_dict in node_data.items():
        for config, packets in config_dict.items():
            for packet, data in packets.items():
                if data["received"] == 1:
                    if node not in new_node_data:
                        new_node_data[node] = {}
                    if config not in new_node_data[node]:
                        new_node_data[node][config] = []
                    new_node_data[node][config].append(data["transmissions"])
                else:
                    if node not in new_node_data:
                        new_node_data[node] = {}
                    if config not in new_node_data[node]:
                        new_node_data[node][config] = []
                    new_node_data[node][config].append(0)

    # Output the resulting dictionary
    with open(output_json_file, 'w') as file:
        json.dump(new_node_data, file, indent=4)

    output_data_stats = {}
    for node, config_dict in new_node_data.items():
        res = {}
        # Calculate the mean, std_dev and variance for each configuration
        for config, series in config_dict.items():
            mean = sum(series) / len(series)
            std_dev = (sum([(x - mean) ** 2 for x in series]) / len(series)) ** 0.5
            variance = std_dev ** 2
            res[config] = {
                'mean': mean,
                'std_dev': std_dev,
                'variance': variance
            }
        output_data_stats[node] = res

    # Write JSON output to file
    with open(output_json_file_stats, 'w') as outfile:
        json.dump(output_data_stats, outfile, indent=4)
