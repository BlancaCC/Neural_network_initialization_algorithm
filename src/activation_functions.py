ramp_function_one_value = lambda x : min(1,max(0,x))

def ramp_function(M):
    return [[ramp_function_one_value(x) for x in row] for row in M]
