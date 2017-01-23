# Seating Planner
Seating Planner is a Python script that generates seating plans for multi-day events (such as computational neuroscience summer schools). Given a list of guests, tables, and dates, Seating Planner assigns guests to tables for each date while taking into account guest preferences for tables and the availability and capacity of each table. 

## Basic Usage
To generate a seating plan, run `seating_planner.py` on an input file e.g. `my_input_file.txt`, from the command line:

```
python seating_planner.py my_input_file.txt
```

The resulting schedule is written to `my_input.schedule.list.txt`. A sample input file is provided in the repository as `input.txt`.
## The Input File
The information that Seating Planner needs to generate a schedule is provided by the **input file**. The input file is a plain text file that provides information about the guests, tables, and dates, and the various parameters of the schedule. The format of the input file is described below.

### Empty Lines and Comments
Empty lines are ignored. All lines beginning with `#` are treated as comments and ignored.

### The Guest List
The list of guests is provided as one or more lines of comma-separated names, within a `GUESTS` block. For example

```
BEGIN_GUESTS
Alice, Bob, Chris
Diana, Emily, Fred
END_GUESTS
```
adds the guests Alice, Bob, ... , Fred to the list. 

### The Tables List
The list of tables is provided in the same way as the list of guests, within a `TABLES` block. For example

```
BEGIN_TABLES
Cajal, Golgi, Hodgkin
END_TABLES
```
adds the tables Cajal, Golgi and Hodgkin to the list of tables. 

### The Dates List
The list of dates is provided in the same way, e.g.

```
BEGIN_DATES
Monday, Tuesday, Wednesday, Thursday, Friday
END_DATES
```

adds the dates Monday, Tuesday and Wednesday to the list of dates. 

### Guest Preferences
The preferences, if any, of the guests for the various tables are provided within a `GUEST_PREFERENCES` block. Each line specifies the preferences of one guest for each table as a list of comma-separated numbers, one for each table, in the order that tables where specified in the `TABLES` block. E.g. 

```
BEGIN_GUEST_PREFERENCES
Alice  : 1,0,1 
END_GUEST_PREFERENCES
```
states that Alice would like to sit at the Cajal and Hodgkin tables, but not at Golgi. 
Preferences need not be provided for every (or any) guest, but if provided, the preferences of the guest for each table must be specified.

### Table Availablities
Table availabilities are provided on separate lines, one per table, as comma-separated lists of integers, with one entry per date, in the order that dates were specified in the `DATES` block. The value of each entry specifies availablity as follows:

- 0: Table unavailable.
- 1: Table available with default maximum occupancy (see [Parameters](#parameters)).
- *n*: Table available with *n* seats.

For example, the block

```
BEGIN_TABLE_AVAILABILITIES
Cajal:    1,1,0,1,4
END_TABLE_AVAILABILITIES
```
indicates that the table *Cajal* is unavailable on Wednesday, and on Friday can only seat at most 4 guests, rather than the default number specified in the [Parameters](#parameters).
Tables with unspecified availabilities assume the default value for all dates.

### Parameters
Generating schedules requires the following parameters to be specified:

1. `max_guests_per_table_per_date`: The default maximum number of guests per table. 
2. `min_guests_per_table_per_date`: The minimum number of guests per (available) table per date.
3. `max_each_table_per_guest`: The maximum number of times (computed over all dates) each guest should sit at each table.
4. `min_each_table_per_guest`: The minimum number of times a guest should sit at a table they expressed interest in.
5. `occupancy_variance_weight`: How much the variance in table occupancies should be weighted vs. matching guests' preferences. Setting this value higher will tend to make the tables more evenly filled, at the cost of ignoring guests' preferences (see [Algorithm](#algorithm))
6. `random_seed`: The random seed to use when rounding the optimization result (see [Algorithm](#algorithm))
7. `num_rounding_iters`: The number of random rounded schedules to generate before choosing the best one (see [Algorithm](#algorithm)).

Parameters are provided in the `PARAMS` block as consecutive `param = value` lines. All parameters must be specified. An example specification is below:

```
BEGIN_PARAMS
max_guests_per_table_per_date = 10
min_guests_per_table_per_date = 4
max_each_table_per_guest = 2
min_each_desired_table_per_guest = 0.4
occupancy_variance_weight = 10.
random_seed = 0
num_rounding_iters = 1000
END_PARAMS
```

## Algorithm
### The schedule representation 
A schedule is binary multi-dimensional array *X*, where *X[i,j,k]* is equal to 1 if guest *i* is sitting on date *j* at table *k*. We work with the schedule in its vectorized (flattened) form *x*. 
### The loss function
The quality of a schedule is defined as its correlation *c'x* with with the guests' preferences matrix, suitably expanded and flattened to yield a vector *c*. To avoid having very unbalanced tables, we also add a term that penalizes the table occupacy variance. This yields the loss function that we want to minimize:  *L(x) = -c'x + alpha * var(table_occupancy(x))*, where *alpha* is the `occupancy_variance_weight` defined in the [Parameters](#parameters) section.
### The relaxed schedule
If we insist that the values in *x* be binary, minimizing the loss function can become computationally intractable. So instead, we *relax* their domain to the unit interval [0,1]. We can then minimize the loss over this domain subject to linear constraints enforcing that minimum and maximum table occupancies on each date and the number of times each guest should sit at each table. This is a *convex* problem which will, if it's solvable at all (certain settings of the constraints will render the problem unsolvable), will have a unique minimum, which we call the *relaxed schedule*.
### Rounding the relaxed solution
The relaxed schedule looks like a real schedule except that the values of its elements can take on any values between 0 and 1. The process of converting these real values to binary is called rounding. To round our solution we interpret the values in the relaxed solution as the probability that a guest should be seated at a particular table on a given date. We then use these probabilities to generate `num_rounding_iters` random schedules, and pick the one with the lowest loss. This is the *rounded schedule*.
### Greedy reassignment
Finally, we perform a greedy reassignment step where we iteratively consider moving guests form one table to another and repeatedly perform the move that reduces the loss the most, until no better moves can be found. This is the *final schedule*.
## History
This script was written at the [isiCNI2017](http://isicni.gatsby.ucl.ac.uk/) computational neuroscience summer school in Cape Town, South Africa to seat students at dinner with visiting faculty.