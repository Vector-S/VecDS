import time

def report(msg,tic=None,print_out=True):
    length = 100
    report = "\n"+">>>>  "+ msg + " "
    if tic:
        report += '-' * (100 - len(report))
        report +=" TC:{0:.4g}".format(time.time()-tic)
    report+="\n"
    if print_out:
        print(report)



class Report:
    def __init__(self):
        total_time_cost = 0