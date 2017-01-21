import sys
import os as os
from numpy import *
import cvxpy as cvx
from tqdm import tqdm

verbose = True

schedule_name = ""

## The basic lists specifying the problem domain
guests = list()
tables = list()
dates  = list()
availabilities = [] # A num_tables x num_dates binary matrix of table availabilities
preferences    = [] # A num_guests x num_tables matrix of guest preferences for tables

## PARAMETERS
max_guests_per_table_per_date    = None 
min_guests_per_table_per_date    = None
max_each_table_per_guest         = None # The maximum number of dates each guest should sit at each table
min_each_desired_table_per_guest = None # Minimum number of dates a guest should sit at a table they're interested in
occupancy_variance_weight        = None # Weighting of occupancy variance in the loss function. 
random_seed                      = None # Random seed
num_rounding_iters               = None # Number of randomnly generated rounded schedules

## CONSTRAINTS MATRICES
## The rows in each of these matrices when applied to a potential schedule vector x
## will compute the sum of the schedule vector for subsets of guests,tables, and dates.
## The naming convention: A_x__y means the variable x is being summed over,
## for (guest, table, date) configurations that match y.
A_tables__available_tables   = []  
A_tables__unavailable_tables = []
A_guests                     = []
A_guests__available_tables   = []  
A_dates__available_tables    = []  
A_dates__interested_guests_available_tables = [] 

## CONSTRAINT VALUES
## One for each of the constraint matrices, and will store
## the constraint on the values of the sums computed.
b_tables__available_tables   = []
b_tables__unavailable_tables = []
b_guests                     = []
b_guests__available_tables   = []
b_dates__available_tables    = []
b_dates__interested_guests_available_tables = []

## CONVEX PROGRAMMING VARIABLES
c    = [] # The objective, a suitably reshaped version of the guests' preferences
A_eq = [] # The matrix specifying the elements invovled in the equalities
b_eq = [] # The values of the equalities
A_ub = [] # The matrix specifying the elements involved in the upper bounds
b_ub = [] # The values of the upper bounds
## Table occupancy variance variables
mean_table_occupancy_per_date = []
A_table_occupancy             = []

def LOG(string):
    if verbose:
        print string

def parse_input_file(input_file):    
    ### Parses the specified input file to get the
    ### list of guests, tables, dates, preferences,
    ### and the parameters for the optimization
    global schedule_name    
    global guests, tables, dates
    global preferences, availabilities
    global max_guests_per_table_per_date
    global min_guests_per_table_per_date
    global max_each_table_per_guest
    global min_each_desired_table_per_guest
    global occupancy_variance_weight
    global random_seed
    global num_rounding_iters

    parts = os.path.basename(input_file).split(".")
    schedule_name = parts[0]
    LOG("SCHEDULE NAME: {}\n".format(schedule_name))

    ## Dictionaries mapping names to indices.
    GUEST2ID= []
    TABLE2ID= []
    DATE2ID = []
    
    state = "SCAN"
    with open(input_file,"r") as fp:
        for line in fp:
            line = line.strip()
            if len(line)==0:
                continue

            if line[0] == "#": # Comment line
                continue
    
            old_state = copy(state)
            if state == "SCAN":
                line_to_state = {
                    "BEGIN_GUESTS":"READ_GUESTS",
                    "BEGIN_TABLES":"READ_TABLES",
                    "BEGIN_DATES" :"READ_DATES",
                    "BEGIN_GUEST_PREFERENCES"   :"READ_GUEST_PREFERENCES",
                    "BEGIN_TABLE_AVAILABILITIES":"READ_TABLE_AVAILABILITIES",
                    "BEGIN_PARAMS":"READ_PARAMS"}
                state = line_to_state[line]
                if state == "READ_GUEST_PREFERENCES":
                    preferences = zeros((len(guests), len(tables)))
                elif state == "READ_TABLE_AVAILABILITIES":
                    availabilities = ones((len(tables), len(dates)))

            if old_state != state:
                LOG("'{}' caused {} => {}".format(line, old_state, state))
                continue

            if state == "READ_PARAMS":
                if line == "END_PARAMS":
                    state = "SCAN"
                else:
                    name_value = line.split("=")
                    param = name_value[0].strip()
                    value = float(name_value[1].strip())
                    exec(line, globals()) # Set the value of the parameter
                    LOG("Set parameter {} to {}.".format(param,eval(param)))
                
            if state == "READ_GUESTS":
                if line == "END_GUESTS":
                    GUEST2ID = dict(zip(guests, range(len(guests))))
                    state = "SCAN"
                else:
                    new_guests = [s.strip() for s in line.split(",")]
                    new_guests = [g for g in new_guests if len(g)>0]
                    guests = guests + new_guests
                    LOG("Appended {} guests: {}".format(len(new_guests), new_guests))
    
            if state == "READ_TABLES":
                if line == "END_TABLES":
                    TABLE2ID = dict(zip(tables, range(len(tables))))
                    state = "SCAN"
                else:
                    new_tables = [s.strip() for s in line.split(",")]
                    new_tables = [t for t in new_tables if len(t)>0]                    
                    tables = tables + new_tables
                    LOG("Appended {} tables: {}".format(len(new_tables), new_tables))            
    
            if state == "READ_DATES":
                if line == "END_DATES":
                    DATE2ID = dict(zip(dates, range(len(dates))))
                    state = "SCAN"
                else:
                    new_dates = [s.strip() for s in line.split(",")]
                    new_dates = [d for d in new_dates if len(d)>0]
                    dates = dates + new_dates
                    LOG("Appended {} dates: {}".format(len(new_dates), new_dates))
    
            if state == "READ_GUEST_PREFERENCES":
                if line == "END_GUEST_PREFERENCES":
                    state = "SCAN"
                else:
                    vals = line.split(":")
                    guest = vals[0].strip()
                    prefs = [int(s) for s in vals[1].strip().split(",")]
                    preferences[GUEST2ID[guest], :] = prefs
    
            if state == "READ_TABLE_AVAILABILITIES":
                if line == "END_TABLE_AVAILABILITIES":
                    state = "SCAN"
                else:
                    vals = line.split(":")
                    table = vals[0].strip()
                    avail = [int(s) for s in vals[1].strip().split(",")]
                    availabilities[TABLE2ID[table], :] = avail
            
            if old_state != state:
                LOG("'{}' caused {} => {}".format(line, old_state, state))
                if state == "SCAN": # Insert a line break if we're back to scanning mode.
                    LOG("")

def build_marginalization_matrices():
    ### Fills in the values of the marginalization matrices and corresponding vectors.
    ### The rows of the matrices are filled in by flattening a 3D guests x dates x tables
    ### tensor with 1's at the desired locations for each sum. Multiplying each row
    ### with a schedule vector performs the corresponding marginalization.
    ### The vectors are just conformable placeholders that are initailized to 1s.
    
    ## Constraint matrices
    global A_tables__available_tables
    global A_tables__unavailable_tables
    global A_guests
    global A_guests__available_tables 
    global A_dates__available_tables
    global A_dates__interested_guests_available_tables

    ## Constraint values
    global b_tables__available_tables
    global b_tables__unavailable_tables
    global b_guests
    global b_guests__available_tables
    global b_dates__available_tables
    global b_dates__interested_guests_available_tables

    ## Table occupancy variance variables
    global mean_table_occupancy_per_date
    global A_table_occupancy
    
    a0 = zeros((num_guests, num_dates, num_tables))
    
    a_tables__available_tables = zeros((num_guests, num_dates, schedule_length))
    for g in range(num_guests):
        for d in range(num_dates):
            a = copy(a0)
            a[g,d,:] = availabilities[:,d]>0
            a_tables__available_tables[g,d,:] = reshape(a,(schedule_length,),order="C")
    A_tables__available_tables = reshape(a_tables__available_tables,(num_guests*num_dates, schedule_length), order="C")
    b_tables__available_tables = A_tables__available_tables[:,0]*0 + 1    
    
    a_tables__unavailable_tables = zeros((num_guests, num_dates, schedule_length))
    for g in range(num_guests):
        for d in range(num_dates):
            a = copy(a0)
            a[g,d,:] = availabilities[:,d]<0.5
            a_tables__unavailable_tables[g,d,:] = reshape(a,(schedule_length,),order="C")
    A_tables__unavailable_tables = reshape(a_tables__unavailable_tables,(num_guests*num_dates, schedule_length), order="C")
    b_tables__unavailable_tables = A_tables__unavailable_tables[:,0]*0 + 1

    # Keep a_guests__available_tables as num_dates x num_tables x schedule_length
    # The table variance calculations assume it's in this order.  
    a_guests__available_tables = zeros((num_dates, num_tables, schedule_length))
    for t in range(num_tables):
        for d in range(num_dates):
            a = copy(a0)
            a[:,d,t] = availabilities[t,d]>0
            a_guests__available_tables[d,t,:] = reshape(a,(schedule_length,),order="C")
    A_guests__available_tables = reshape(a_guests__available_tables,(num_tables*num_dates, schedule_length), order="C")
    b_guests__available_tables = A_guests__available_tables[:,0]*0 + 1

    a_guests = zeros((num_dates, num_tables, schedule_length))
    for t in range(num_tables):
        for d in range(num_dates):
            a = copy(a0)
            a[:,d,t] = 1
            a_guests[d,t,:] = reshape(a,(schedule_length,),order="C")
    A_guests = reshape(a_guests,(num_tables*num_dates, schedule_length), order="C")
    b_guests = A_guests[:,0]*0 + 1
    
    available_tables_per_date = sum(availabilities>0,axis=0).astype(int)
    number_of_tables_per_date = zeros((sum(available_tables_per_date),))
    ind = 0
    for i in range(len(available_tables_per_date)):
        number_of_tables_per_date[range(ind,ind+available_tables_per_date[i])] = available_tables_per_date[i]
        ind = ind + available_tables_per_date[i]
    mean_table_occupancy_per_date = len(guests)/number_of_tables_per_date.astype(float)

    A_table_occupancy = remove_empty_rows(A_guests__available_tables,A_guests__available_tables[:,0])[0]

    a_dates__available_tables = zeros((num_guests, num_tables, schedule_length))
    for g in range(num_guests):
        for t in range(num_tables):
            a = copy(a0)
            a[g,:,t] = availabilities[t,:]>0
            a_dates__available_tables[g,t,:] = reshape(a,(schedule_length,),order="C")
    A_dates__available_tables = reshape(a_dates__available_tables,(num_guests*num_tables, schedule_length), order="C")
    b_dates__available_tables = A_dates__available_tables[:,0]*0 + 1
    
    a_dates__interested_guests_available_tables = zeros((num_guests, num_tables, schedule_length))
    for g in range(num_guests):
        for t in range(num_tables):
            a = copy(a0)
            a[g,:,t] = (preferences[g,t]>0)*(availabilities[t,:]>0)
            a_dates__interested_guests_available_tables[g,t,:] = reshape(a,(schedule_length,),order="C")
    A_dates__interested_guests_available_tables = reshape(a_dates__interested_guests_available_tables,(num_guests*num_tables, schedule_length), order="C")
    b_dates__interested_guests_available_tables = A_dates__interested_guests_available_tables[:,0]*0 + 1

def remove_empty_rows(A,b):
    ### Finds the rows of A that are all zeros and removes them from A and b.
    row_sums = sum(abs(A)>1e-12,axis=1)
    A1 = A[row_sums>0,:]
    b1 = b[row_sums>0]
    return A1,b1

def build_feasability_matrices():
    ### Combines the constraints matrices and sets the values of the corresponding
    ### constraints vectors to set up the equality and upper bounds constraints for the
    ### linear program.
    
    global A_eq, b_eq
    global A_ub, b_ub
    global c

    ## Build the objective vector
    ## The preferences are for tables, not for dates, so expand to fill in the values
    ## for the dates, and flatten.
    c = reshape(kron(ones((1,num_dates)), preferences), (schedule_length,), order="C")
    
    ## Each guest must be at most at one table each day.
    #  This means that the sum over available tables / guest/day = 1...
    A_eq1 = copy(A_tables__available_tables) 
    b_eq1 = copy(b_tables__available_tables)*1
    
    # ...And the sum over *unavailable* tables / guest / day = 0
    A_eq2 = copy(A_tables__unavailable_tables)
    b_eq2 = copy(b_tables__unavailable_tables)*0
    
    ## Each tables must be have at most n guests each day.
    # This means that the sum over guests/day/available tables <= n
    # n is specified in the availabilities table.
    # n = 0: Unavailable
    # n = 1: max_guests_per_table_per_date
    # n > 1: User specified availability
    A_ub1 = copy(A_guests)
    b_ub1 = reshape(availabilities.T, (num_tables*num_dates,))
    b_ub1[b_ub1==1] = max_guests_per_table_per_date
    
    ## Each tables must have at least n guests each day.
    # This means that the sum over guests/day/available tables  >= n
    A_ub2 = -copy(A_guests__available_tables)
    b_ub2 = -copy(b_guests__available_tables)*min_guests_per_table_per_date
    
    ## Each guest should sit at each table at most n times.
    # This means that the sum over dates/guest/available tables <= n
    A_ub3 = copy(A_dates__available_tables)
    b_ub3 = copy(b_dates__available_tables)*max_each_table_per_guest
    
    ## Each guest should sit with each tables they want to chat to at least n times.
    # We soften this to the sum over dates/interested guest/available tables >= n
    A_ub4 = -copy(A_dates__interested_guests_available_tables)
    b_ub4 = -copy(b_dates__interested_guests_available_tables)*min_each_desired_table_per_guest

    ## Combine and cleanup the equalities
    A_eq = concatenate((A_eq1, A_eq2), axis=0)
    b_eq = concatenate((b_eq1, b_eq2), axis=0)
    A_eq, b_eq = remove_empty_rows(A_eq,b_eq)

    ## Combine and clean up the upper bounds
    A_ub = concatenate((A_ub1, A_ub2, A_ub3, A_ub4), axis=0)
    b_ub = concatenate((b_ub1, b_ub2, b_ub3, b_ub4), axis=0)
    A_ub, b_ub = remove_empty_rows(A_ub, b_ub)

    LOG("\tNumber of variables: {}".format(len(c)))
    LOG("\tNumber of upper-bound constraints: {}".format(A_ub.shape[0]))
    LOG("\tNumber of equality constraints: {}".format(A_eq.shape[0]))
    
def run_convex_program():
    x = cvx.Variable(schedule_length)
    objective = cvx.Minimize(-c.T * x + 0.5*occupancy_variance_weight*cvx.sum_squares(A_table_occupancy*x - mean_table_occupancy_per_date)/A_table_occupancy.shape[0])
    constraints  = [x>=0,
                    x<=1,
                    A_eq*x == b_eq,
                    A_ub*x <= b_ub]
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    print "\tstatus: {}".format(prob.status)
    print "\toptimal relaxed objective: {:.2f}".format(prob.value)
    if prob.value is Inf:
        raise Exception("Problem is infeasable. Stopping.")
    relaxed_schedule = reshape(array(x.value), (schedule_length,))
    return relaxed_schedule

def round_solution(relaxed_solution, num_rounds = 100, mode = "random"):
    ### The linear programming solution returns
    ### values in [0,1] for the seating assignments.
    ### We generate a binary schedule by treating these as probabilities
    ### and randomly assigning guests to tables according to their
    ### probability vectors over tables on a given date.
    num_rounds    = {"random":num_rounds, "max":1}[mode]
    best_schedule = copy(relaxed_solution)
    best_loss     = Inf
    losses        = zeros((num_rounds,))
    for r in tqdm(range(num_rounds)):
        schedule = 0*relaxed_solution
        for i in range(0, schedule_length, num_tables):
            p = relaxed_schedule[range(i,i+num_tables)]
            p[p<=0] = 1e-12 # Set to small non-zero value so we can safely sum.
            p = p/sum(p)
            which_table = {"random":random.choice(num_tables, p = p), "max":argmax(p)}[mode]
            schedule[i + which_table] = 1

        losses[r] = loss_function(schedule)

        if losses[r] < best_loss:
            best_loss = losses[r]
            best_schedule = copy(schedule)
    
    return best_schedule, losses

def config_from_ind(i):
    table = mod(i, num_tables)
    date  = mod((i - table)/num_tables, num_dates)
    guest = (i - table - num_tables*date)/num_tables/num_dates
    return (table, date, guest)

def ind_from_config(table, date, guest):
    return table + date*num_tables + guest*num_tables*num_dates

def loss_function(s):
    return -dot(c.T, s) + occupancy_variance_weight*0.5*mean((dot(A_table_occupancy,s) - mean_table_occupancy_per_date)**2)

def greedy_optimize_rounded_solution(rounded_solution, max_iters = 10000):
    best_loss = loss_function(rounded_solution)
    best_schedule = copy(rounded_solution)
    for t in range(max_iters):
        seating_plan = where(best_schedule)[0]
        schedule = copy(best_schedule)
        descr = "No change."
        improved = False
        for old_ind in seating_plan:
            table, date, guest = config_from_ind(old_ind)
            available_tables = list(where(availabilities[:,date])[0])
            available_tables.remove(table)
            for new_table in available_tables:
                new_ind      = ind_from_config(new_table,date,guest)
                new_schedule = copy(schedule)
                new_schedule[old_ind] = 0
                new_schedule[new_ind] = 1
                # Check that the new schedule is feasable
                feasable = allclose(dot(A_eq,new_schedule), b_eq) and all(dot(A_ub, new_schedule) <= b_ub)
                # Check that the loss is lower
                new_loss   = loss_function(new_schedule)
                lower_loss = new_loss < best_loss
                if lower_loss and feasable:
                    improved = True
                    descr = "Moving {} from {} to {} on {} improved the loss from {} to {}".format(t, guests[guest], tables[table], tables[new_table], date, best_loss, new_loss)
                    best_loss = new_loss
                    best_schedule = copy(new_schedule)
        LOG("Iteration {}: {}".format(t,descr))
        if not improved:
            break
    return best_schedule

def write_schedule_as_list(schedule_as_matrix):
    file_name = "{}.schedule.list.txt".format(schedule_name)
    with (open(file_name ,"w")) as fp:
        ind = 0
        for d in range(num_dates):
            fp.write("{}\n".format(dates[d].upper()))
            for t in range(num_tables):
                ind = (d*num_tables + t)
                col = schedule_as_matrix[:,ind]
                guests_list = [guests[i] for i in range(num_guests) if schedule_as_matrix[i,ind]>0]
                scores_list   = [preferences[i,t] for i in range(num_guests) if schedule_as_matrix[i,ind]>0]
                if len(guests_list) > 0:
                    fp.write("\t{}\n".format(tables[t]))
                    for i in range(len(guests_list)):
                        fp.write("\t\t{:16s} ({})\n".format(guests_list[i], scores_list[i]))
    LOG("\tWrote {}.".format(file_name))

### THE DRIVER CODE

if len(sys.argv)!=2:
    print "Usage: seating_plan input.txt"
    exit()
else:
    input_file = sys.argv[1]

LOG("Parsing {}.".format(input_file))
parse_input_file(input_file)
    
num_tables = len(tables)
num_guests = len(guests)
num_dates  = len(dates)

schedule_length = num_tables*num_guests*num_dates

random.seed(random_seed)

LOG("BUILDING MARGINALIZATION MATRICES.")
build_marginalization_matrices()

LOG("BUILDING FEASABILITY MATRICES.")
build_feasability_matrices()

LOG("RUNNING CONVEX PROGRAM.")
relaxed_schedule = run_convex_program()

LOG("ROUNDING SOLUTION.")
rounded_schedule, losses = round_solution(relaxed_schedule, num_rounds = num_rounding_iters, mode="random")
LOG("\tObjective for rounded schedule: {:.2f}".format(loss_function(rounded_schedule)))
LOG("GREEDY OPTIMIZING ROUNDED SOLUTION.")
schedule = greedy_optimize_rounded_solution(rounded_schedule)
LOG("\tObjective for final schedule:   {:.2f}".format(loss_function(schedule)))
schedule_as_matrix = reshape(schedule, (num_guests, num_tables*num_dates), order="C")
LOG("WRITING SCHEDULE AS LIST.")
write_schedule_as_list(schedule_as_matrix)
LOG("DONE.")
