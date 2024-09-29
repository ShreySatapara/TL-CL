# create a function that load results from folder and print forward transfer, backward transfer, average performance, and later generalization performance
from prettytable import PrettyTable



def print_argparse_args(parser):
    table = PrettyTable()
    table.field_names = ["Argument", "Value"]

    for arg in vars(parser.parse_args()):
        arg_value = getattr(parser.parse_args(), arg)
        table.add_row([arg, arg_value])

    print(table)
    
def print_result_table(result_dict):
    table = PrettyTable()
    table.field_names = ["Task", "Language", "Metric Type", "Score"]
    #print(table)
    for result in result_dict:
        table.add_row([result["task"], result['language'], result['score']['metrics'], result['score']['score']])
    print(table)