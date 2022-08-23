"""
    Read the json output from train.py and convert it to .csv
"""
import json
import sys

def parse_train(data):
    test_ids = []
    out = "training node id,"
    for k in data[0]['total_loss_for_testing_nodes']:
        out += str(k) + ","
        test_ids.append(k)

    out += "max accuracy"
    print(out)

    for row in data:
        out = str(row['leave_out']) + ","
        for node_id in test_ids:
            out += str(row['total_loss_for_testing_nodes'][node_id]) + ","
        out += str(row['max_testing_accuracy'])
        print(out)

def parse_influence(data):
    test_ids = []
    for k in data:
        test_ids.append(k['node_id'])

    print(",".join([str(i) for i in test_ids]))

    total = len(data[0]['influence']['influences'])
    for influence_id in range(total):
        out = ""
        for i in range(len(test_ids)):
            out += str(data[i]['influence']['influences'][influence_id]) + ","
        print(out)
    
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print(f'Usage: {sys.argv[0]} [json file]')
        exit()

    with open(sys.argv[1]) as json_file:
        data = json.load(json_file)

        if 'leave_out' in data[0]:
            parse_train(data)
        else:
            parse_influence(data)
