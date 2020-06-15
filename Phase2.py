"""
-----------------------------------------------------------------------------------------------
Simulation of a Restaurant
Co-written by Mohammadali Allahdadi & Ehsan Razzaghi

No limit on Queue's length
People get service in a FIFO system

Starting State = System is empty
-----------------------------------------------------------------------------------------------
"""

import math
import random
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

# Python console options, used for logging
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Choosing a theme for graphs
sb.set()
sb.set(style="whitegrid")

# Starting state values
STARTING_RECQ = 0  # Number of people in reception queue
STARTING_FOODQ = 0  # Number of people in food queue
STARTING_SALQ = 0  # Number of people in saloon queue

STARTING_RECOP = 0  # Number of working reception operators
STARTING_UNRECOP = 5  # Number of idle reception operators
STARTING_RECREST = False  # Is it time for reception operators' rest?

STARTING_FOODOP = 0  # Number of working food operators
STARTING_UNFOODOP = 2  # Number of idle food operators
STARTING_FOODREST = False  # Is it time for reception operators' rest?

STARTING_SALOP = 30  # Number of empty tables

# Resting times
REST_TIME_FIRST = 50
REST_TIME_SECOND = 110
REST_TIME_THIRD = 230
REST_TIME_FOURTH = 290

TIME_OF_RESTING = 10  # How much time does resting take?

NO_CUSTOMER = '-1'  # For the times when we don't have a customer to pass to a function

SIMULATION_STARTING_TIME = 0  # Clock begins at 0

# Logging Values. Will log if set to 'True'
LOG_DATA = False  # Print all the simulation data in each step
LOG_STEPS = True  # Print all major simulation steps
LOG_EXCEL = True  # Decides if an Excel output is generated
LOG_CHART = False  # Draw a chart output with recQ, foodQ and salQ over time
LOG_REQ = False  # Print Management request at the end of each replication
LOG_WARMUP = False  # Draw a chart output with warmup data
LOG_SENSITIVITY = False

RANDOM_SEED = 4913
random.seed(4913)
if LOG_SENSITIVITY:
    random.seed(RANDOM_SEED)  # Sensitivity Test

WARMUP_STEP = 30  # Warmup logging frequency (in minutes)
warmup_header_list = ['Replication',
                      'Period',
                      'Mean Time Spent in Reception'
                      ]
warmUpData = pd.DataFrame(columns=warmup_header_list)  # Contains warmup data from different replications

replication_header_list = ['Replication',
                           'R1',
                           'R2',
                           'R3_1', 'R3_2',
                           'R4_1', 'R4_2',
                           'R5'
                           ]
replicationData = pd.DataFrame(columns=replication_header_list)  # Contains requested values from each replication

# Global data collectors
gData = dict()
gData['warmup_y'] = []
gData['warmup_x'] = []
gData['times'] = []


def starting_state():
    # State
    state = dict()
    state['recQueue'] = dict()
    state['foodQueue'] = dict()
    state['salQueue'] = dict()

    state['recQ'] = STARTING_RECQ
    state['foodQ'] = STARTING_FOODQ
    state['salQ'] = STARTING_SALQ
    state['recOP'] = STARTING_RECOP
    state['unRecOP'] = STARTING_UNRECOP
    state['recRest'] = STARTING_RECREST
    state['foodOP'] = STARTING_FOODOP
    state['unFoodOP'] = STARTING_UNFOODOP
    state['foodRest'] = STARTING_FOODREST
    state['salOP'] = STARTING_SALOP

    data_collecting = dict()  # Data Collecting Dict, saves the Previous clock, the customer times and so on
    data_collecting['EClock'] = SIMULATION_STARTING_TIME  # The event clock

    # The customer Time values in our system {'Ci':
    # [t0(enter Q1), t1(left Q1), t2(get REC),
    # t3(enter Q2), t4(left Q2), t5(get FOOD),
    # t6(enter Q3), t7(left Q3), t8(end FOOD),
    # t9(END)]}
    data_collecting['Customers'] = dict()

    data_collecting['lastCustomer'] = 1  # Keeps the number of our last entered customer

    # Queue chart data collectors (Number of people in each queue)
    data_collecting['stat_x'] = []
    data_collecting['recQ_stat_y'] = []
    data_collecting['foodQ_stat_y'] = []
    data_collecting['salQ_stat_y'] = []

    data_collecting['totalCustomers'] = np.zeros(shape=10)  # Total number of customers in each state

    # Cumulative statistics
    cumulative_stat = dict()
    cumulative_stat["recQueue Length"] = 0  # for Lrq
    cumulative_stat["recQueue Waiting Time"] = 0  # for Wrq
    cumulative_stat["recOP Busy Time"] = 0  # for Brq

    cumulative_stat["foodQueue Length"] = 0  # for Lfq
    cumulative_stat["foodQueue Waiting Time"] = 0  # for Wfq
    cumulative_stat["foodOP Busy Time"] = 0  # for Bfq

    cumulative_stat["salQueue Length"] = 0  # for Lsq
    cumulative_stat["salQueue Waiting Time"] = 0  # for Wsq
    cumulative_stat["salOP Busy Time"] = 0  # for Bsq

    cumulative_stat["Time in System"] = 0  # Time spent in the system

    future_event_list = list()

    # Create the entrance of our first customer
    FEL_maker(future_event_list, 'recQueue', SIMULATION_STARTING_TIME, 'C1')

    # Car and Bus Entrance
    if addBus:
        FEL_maker(future_event_list, 'recQueueBus', SIMULATION_STARTING_TIME, NO_CUSTOMER)
    FEL_maker(future_event_list, 'recQueueCar', SIMULATION_STARTING_TIME, NO_CUSTOMER)

    if not LOG_WARMUP:
        # Reception crew resting
        FEL_maker(future_event_list, 'startRecRest', REST_TIME_FIRST, NO_CUSTOMER)
        FEL_maker(future_event_list, 'startRecRest', REST_TIME_SECOND, NO_CUSTOMER)
        FEL_maker(future_event_list, 'startRecRest', REST_TIME_THIRD, NO_CUSTOMER)
        FEL_maker(future_event_list, 'startRecRest', REST_TIME_FOURTH, NO_CUSTOMER)

        # Food crew resting
        FEL_maker(future_event_list, 'startFoodRest', REST_TIME_FIRST, NO_CUSTOMER)
        FEL_maker(future_event_list, 'startFoodRest', REST_TIME_SECOND, NO_CUSTOMER)
        FEL_maker(future_event_list, 'startFoodRest', REST_TIME_THIRD, NO_CUSTOMER)
        FEL_maker(future_event_list, 'startFoodRest', REST_TIME_FOURTH, NO_CUSTOMER)

    return state, data_collecting, future_event_list, cumulative_stat


def FEL_maker(future_event_list, event_type, clock, customer):
    if event_type == "recQueue":
        event_time = clock + t_user()
    elif event_type == "recQueueCar":
        event_time = clock + t_car()
    elif event_type == "recQueueBus":
        event_time = clock + t_bus()
    elif event_type == "getRec":
        event_time = clock + t_getRec()
    elif event_type == "startRecRest":
        event_time = clock
    elif event_type == "endRecRest":
        event_time = clock + TIME_OF_RESTING
    elif event_type == "foodQueue":
        event_time = clock + delay('recMove')
    elif event_type == "getFood":
        event_time = clock + t_getFood()
    elif event_type == "startFoodRest":
        event_time = clock
    elif event_type == "endFoodRest":
        event_time = clock + TIME_OF_RESTING
    elif event_type == "salQueue":
        event_time = clock + delay('foodMove')
    elif event_type == "endFood":
        event_time = clock + t_endFood()
    elif event_type == "exitSys":
        event_time = clock + delay('salMove')
    else:
        event_time = 0

    # additional element in event notices (Customer No.)
    new_event = {'Event Type': event_type,
                 'Event Time': round(event_time, 3), 'Customer': customer}
    future_event_list.append(new_event)


def t_user():
    return randTime_exp(3)


def t_car():
    return randTime_exp(5)


def t_bus():
    return randTime_uniform(60, 180)


def t_getRec():
    return randTime_tri(1, 4, 2) + randTime_tri(1, 3, 2)


def t_getFood():
    return randTime_uniform(0.5, 2)


def t_endFood():
    return randTime_tri(10, 30, 20)


def recQueue(future_event_list, state, data, clock, customer, cum_stat, create_next=True):
    # Create a list for the new customer
    data['Customers'][customer] = []

    # Is the server busy?
    # NO
    if state['unRecOP'] > 0:
        data['Customers'][customer] = [clock, clock]  # log t0, t1
        data['totalCustomers'][0] += 1
        state['recOP'] += 1  # Make an operator busy
        state['unRecOP'] -= 1

        FEL_maker(future_event_list, 'getRec', clock, customer)

    # YES
    else:
        data['Customers'][customer] = [clock]  # log t0
        state['recQ'] += 1  # Add them to the Queue
        state['recQueue'][customer] = clock

    if create_next:
        data['lastCustomer'] += 1
        data['Customers']['C' + str(data['lastCustomer'])] = []
        FEL_maker(future_event_list, 'recQueue', clock, 'C' + str(data['lastCustomer']))  # Create the next entrance
    # cum_stat['recQueue Length'] += state['recQ'] * (clock - data['EClock'])  # Log the number of people in Queue
    advanceTime(data, clock)


def carPassengers():  # Returns the number of car passengers
    rand = random.random()
    if 0.2 > rand >= 0:
        return 1
    elif 0.5 > rand >= 0.2:
        return 2
    elif 0.8 > rand >= 0.5:
        return 3
    return 4


def recQueueCar(future_event_list, state, data, clock, cum_stat):
    passengers = carPassengers()
    for p in range(0, passengers):
        data['lastCustomer'] += 1
        data['Customers']['C' + str(data['lastCustomer'])] = []
        # Send in the car passenger as a normal walking customer, but don't create the next customer based on it.
        recQueue(future_event_list, state, data, clock, 'C' + str(data['lastCustomer']), cum_stat, False)
    FEL_maker(future_event_list, 'recQueueCar', clock, NO_CUSTOMER)  # Predict the next car's arrival
    advanceTime(data, clock)


def recQueueBus(future_event_list, state, data, clock, cum_stat):
    passengers = randTime_poisson(30)
    for p in range(0, passengers):
        data['lastCustomer'] += 1
        data['Customers']['C' + str(data['lastCustomer'])] = []
        # Send in the bus passenger as a normal walking customer, but don't create the next customer based on it.
        recQueue(future_event_list, state, data, clock, 'C' + str(data['lastCustomer']), cum_stat, False)
    advanceTime(data, clock)


def getRec(future_event_list, state, data, clock, customer, cum_stat):
    # Log the time
    data['Customers'][customer].append(clock)  # log t2
    data['totalCustomers'][1] += 1
    cum_stat["recOP Busy Time"] \
        += data['Customers'][customer][2] - data['Customers'][customer][1]

    FEL_maker(future_event_list, "foodQueue", clock, customer)  # Send the customer off

    state['recOP'] -= 1  # Make the operator idle

    if state['recRest']:  # Check if it's time for their rest
        # It's time for resting
        FEL_maker(future_event_list, "endRecRest", clock, NO_CUSTOMER)
        state['recRest'] = False
    else:
        # It's not time for resting
        state['unRecOP'] += 1
        if state['recQ'] > 0:
            # There's someone in the queue. Make the operator busy
            state['recQ'] -= 1
            state['recOP'] += 1
            state['unRecOP'] -= 1

            # Send in the next customer
            firstCustomer = firstInRecQueue(state)
            data['Customers'][firstCustomer].append(clock)  # log t1
            data['totalCustomers'][0] += 1
            cum_stat["recQueue Waiting Time"] \
                += data['Customers'][firstCustomer][1] - data['Customers'][firstCustomer][0]
            FEL_maker(future_event_list, "getRec", clock, firstCustomer)

            del state['recQueue'][firstCustomer]  # Delete the customer from queue
    advanceTime(data, clock)


def foodQueue(future_event_list, state, data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)  # log t3
    data['totalCustomers'][2] += 1

    # Is the server busy?
    if state['unFoodOP'] > 0:
        # NO
        data['Customers'][customer].append(clock)  # log t4
        data['totalCustomers'][3] += 1

        state['foodOP'] += 1  # Make 1 Operator Busy
        state['unFoodOP'] -= 1

        FEL_maker(future_event_list, 'getFood', clock, customer)

    else:
        # YES
        state['foodQ'] += 1
        state['foodQueue'][customer] = clock

    # cum_stat['foodQueue Length'] += state['foodQ'] * (clock - data['EClock'])  # Log the number of people in Queue
    advanceTime(data, clock)


def getFood(future_event_list, state, data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)  # log t5
    data['totalCustomers'][4] += 1
    cum_stat["foodOP Busy Time"] \
        += data['Customers'][customer][5] - data['Customers'][customer][4]

    FEL_maker(future_event_list, "salQueue", clock, customer)  # Send the customer off

    state['foodOP'] -= 1  # Make the operator idle

    # Is it time for resting?
    if state['foodRest']:
        # YES
        FEL_maker(future_event_list, "endFoodRest", clock, NO_CUSTOMER)
        state['foodRest'] = False
    else:
        # NO
        state['unFoodOP'] += 1
        if state['foodQ'] > 0:
            # There's someone in the queue.
            state['foodQ'] -= 1
            state['foodOP'] += 1  # Make the operator busy
            state['unFoodOP'] -= 1

            # Send in the next customer
            firstCustomer = firstInFoodQueue(state)
            data['Customers'][firstCustomer].append(clock)  # log t4
            data['totalCustomers'][3] += 1
            cum_stat["foodQueue Waiting Time"] \
                += data['Customers'][firstCustomer][4] - data['Customers'][firstCustomer][3]
            FEL_maker(future_event_list, "getFood", clock, firstCustomer)

            del state['foodQueue'][firstCustomer]  # Delete the customer from queue
    advanceTime(data, clock)


def salQueue(future_event_list, state, data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)  # log t6
    data['totalCustomers'][5] += 1

    # Are the tables full
    if state['salOP'] > 0:
        # NO
        data['Customers'][customer].append(clock)  # log t7
        data['totalCustomers'][6] += 1
        state['salOP'] -= 1  # Make 1 Table Taken

        FEL_maker(future_event_list, 'endFood', clock, customer)

    else:
        # YES
        state['salQ'] += 1
        state['salQueue'][customer] = clock

    # cum_stat['salQueue Length'] += state['salQ'] * (clock - data['EClock'])
    advanceTime(data, clock)


def endFood(future_event_list, state, data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)  # log t8
    data['totalCustomers'][7] += 1

    FEL_maker(future_event_list, "exitSys", clock, customer)  # Send the customer off

    state['salOP'] += 1  # Set the table free

    if state['salQ'] > 0:
        # There's someone in the queue
        state['salQ'] -= 1  # Make the operator busy
        state['salOP'] -= 1

        # Send in the next customer
        firstCustomer = firstInSalQueue(state)
        data['Customers'][firstCustomer].append(clock)  # log t7
        data['totalCustomers'][6] += 1
        cum_stat["salQueue Waiting Time"] \
            += data['Customers'][firstCustomer][7] - data['Customers'][firstCustomer][6]
        FEL_maker(future_event_list, "endFood", clock, firstCustomer)

        # Delete the customer from queue
        del state['salQueue'][firstCustomer]
    advanceTime(data, clock)


def exitSys(data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)  # log t9
    data['totalCustomers'][8] += 1
    cum_stat["Time in System"] \
        += data['Customers'][customer][9] - data['Customers'][customer][0]
    advanceTime(data, clock)


def startRecRest(future_event_list, state, data, clock):
    if state['unRecOP'] > 0:
        state['unRecOP'] -= 1
        FEL_maker(future_event_list, 'endRecRest', clock, NO_CUSTOMER)
    else:
        state['recRest'] = True
    advanceTime(data, clock)


def startFoodRest(future_event_list, state, data, clock):
    if state['unFoodOP'] > 0:
        state['unFoodOP'] -= 1
        FEL_maker(future_event_list, 'endFoodRest', clock, NO_CUSTOMER)
    else:
        state['foodRest'] = True
    advanceTime(data, clock)


def endRecRest(future_event_list, state, data, clock, cum_stat):
    if state['recQ'] > 0:  # Is there someone in the queue?
        # YES
        state['recQ'] -= 1  # Make the operator busy
        state['recOP'] += 1

        # Send in the first customer in the queue
        firstCustomer = firstInRecQueue(state)
        FEL_maker(future_event_list, 'getRec', clock, firstCustomer)
        data['Customers'][firstCustomer].append(clock)  # log t1
        data['totalCustomers'][0] += 1
        cum_stat["recQueue Waiting Time"] \
            += data['Customers'][firstCustomer][1] - data['Customers'][firstCustomer][0]
        del state['recQueue'][firstCustomer]
    else:
        # NO
        state['unRecOP'] += 1
    advanceTime(data, clock)


def endFoodRest(future_event_list, state, data, clock, cum_stat):
    if state['foodQ'] > 0:  # Is there someone in the queue?
        # YES
        state['foodQ'] -= 1  # Make the operator busy
        state['foodOP'] += 1

        # Send in the first customer in the queue
        firstCustomer = firstInFoodQueue(state)
        FEL_maker(future_event_list, 'getFood', clock, firstCustomer)
        data['Customers'][firstCustomer].append(clock)  # log t4
        data['totalCustomers'][3] += 1
        cum_stat["foodQueue Waiting Time"] \
            += data['Customers'][firstCustomer][4] - data['Customers'][firstCustomer][3]
        del state['foodQueue'][firstCustomer]
    else:
        # NO
        state['unFoodOP'] += 1
    advanceTime(data, clock)


def firstInRecQueue(state):  # Returns the first customer in the reception queue
    firstCustomer = min(state['recQueue'], key=lambda k: state['recQueue'][k])
    return firstCustomer


def firstInFoodQueue(state):  # Returns the first customer in the food queue
    firstCustomer = min(state['foodQueue'], key=lambda k: state['foodQueue'][k])
    return firstCustomer


def firstInSalQueue(state):  # Returns the first customer in the saloon queue
    firstCustomer = min(state['salQueue'], key=lambda k: state['salQueue'][k])
    return firstCustomer


# uniform dist
def randTime_uniform(a, b):
    return (b - a) * random.random() + a


# exp dist
def randTime_exp(mean):
    return math.log(random.random()) / (-1 / mean)


# Triangular dist
# https://en.wikipedia.org/wiki/Triangular_distribution#Generating_triangular-distributed_random_variates
def randTime_tri(a, b, c):
    # a->start b->end c->mode
    F_C = (c - a) / (b - a)
    newRandom = random.random()
    if F_C > newRandom >= 0:
        return a + math.sqrt(newRandom * (b - a) * (c - a))
    return b - math.sqrt((1 - newRandom) * (b - a) * (b - c))


# Poisson dist
def randTime_poisson(mean):
    n = 0
    P = 1
    e_alpha = math.e ** (-1 * mean)
    while True:
        newRandom = random.random()
        P = P * newRandom
        if P < e_alpha:
            # Accepted
            return n
        else:
            # Rejected
            n = n + 1


def delay(name):
    if name == "recMove" or name == "foodMove":
        return randTime_exp(0.5)
    elif name == "salMove":
        return randTime_exp(1)
    return 0


def advanceTime(data, clock):
    data['EClock'] = clock


def output_excel_pandas(future_event_list, state, row_num, cumulative_stat):
    global output_df  # Use the dataframe global variable
    global max_fel  # Use the max_fel global variable
    future_event_list = sorted(future_event_list, key=lambda x: x['Event Time'])
    # This make the calculations much less
    new_row = [row_num, future_event_list[0]['Event Type'], future_event_list[0]['Event Time'],  # Step, Current, Clock
               state['recQ'], state['recOP'], state['unRecOP'], state['recRest'],  # Reception
               state['foodQ'], state['foodOP'], state['unFoodOP'], state['foodRest'],  # Food
               state['salQ'], state['salOP'],  # Saloon
               cumulative_stat["recQueue Length"],
               cumulative_stat["recQueue Waiting Time"],
               cumulative_stat["recOP Busy Time"],

               cumulative_stat["foodQueue Length"],
               cumulative_stat["foodQueue Waiting Time"],
               cumulative_stat["foodOP Busy Time"],

               cumulative_stat["salQueue Length"],
               cumulative_stat["salQueue Waiting Time"],
               cumulative_stat["salOP Busy Time"],

               cumulative_stat["Time in System"]
               ]

    # if we pass the max_fel size
    if len(future_event_list) - 1 > max_fel:
        # loop through the sorted fel to add the the new columns and their value (all of them but the current event)
        for fel_counter in range(max_fel, len(future_event_list) - 1):
            # create the new columns
            output_df['Future Event Type ' + str(fel_counter + 1)] = ""
            output_df['Future Event Time ' + str(fel_counter + 1)] = ""
            fel_counter += 1
        max_fel = len(future_event_list) - 1  # new max_fel

    else:
        # to match the length of row and columns
        for counter in range(max_fel - len(future_event_list) + 1):
            future_event_list.append({'Event Type': "", 'Event Time': ""})
        # add needed values to new row
    for fel in future_event_list[1:]:
        new_row.extend((fel['Event Type'], fel['Event Time']))

    # add the row at the end of the output dataframe
    output_df.loc[len(output_df)] = new_row  # Search it on Google


def excel_formatting_pandas():
    excel_output = pd.ExcelWriter("Output_Pandas.xlsx", engine='xlsxwriter')  # create an excel file
    output_df.to_excel(excel_output, sheet_name='Restaurant-Simulation', index=False)  # pour down the data in the excel
    workbook = excel_output.book  # change the excel to a workbook object
    worksheet = excel_output.sheets['Restaurant-Simulation']  # take the wanted sheet

    # format for the header. Check out the website:
    cell_format_header = workbook.add_format()
    cell_format_header.set_align('center')
    cell_format_header.set_align('vcenter')
    cell_format_header.set_font('Times New Roman')
    cell_format_header.set_bold(True)
    worksheet.set_row(0, None, cell_format_header)

    # align the whole excel to the center (I did this way may be there are other ways)
    worksheet.set_column(0, 0, 5)
    worksheet.set_column(1, 1, 13)
    worksheet.set_column(2, 2, 9)
    worksheet.set_column(3, 13, 20)
    worksheet.set_column(14, 23, 27)
    worksheet.set_column(5, 4 + 2 * max_fel, 19)
    cell_format = workbook.add_format()
    cell_format.set_align('center')
    for row in range(len(output_df)):
        worksheet.set_row(row + 1, None, cell_format)

    # Save the excel
    excel_output.save()


def simulation(simulation_time, replication_number, replication_data, warmup_data, total_replication):
    if LOG_SENSITIVITY:
        random.seed(4913)  # Sensitivity Test
    # Max number of people in Food Queue
    maxFoodQ = 0
    maxSalQ = 0
    # Row of excel file
    row_num = 1
    # Time Period (for warm-up period detection)
    timePeriod = 0

    state, data, future_event_list, cumulative_stat = starting_state()  # Starting state
    clock = 0

    # Add the EoS event
    future_event_list.append({'Event Type': 'End of Simulation', 'Event Time': simulation_time})

    # Continue till the simulation time ends
    while clock < simulation_time:

        sorted_fel = sorted(future_event_list, key=lambda x: x['Event Time'])  # Sort the FEL based on event times
        current_event = sorted_fel[0]  # The first element is what happening now
        clock = current_event['Event Time']  # Move the time forward

        cumulative_stat['recQueue Length'] += state['recQ'] * (clock - data['EClock'])
        cumulative_stat['foodQueue Length'] += state['foodQ'] * (clock - data['EClock'])
        cumulative_stat['salQueue Length'] += state['salQ'] * (clock - data['EClock'])

        if clock < simulation_time:
            current_customer = current_event['Customer']
            if current_event['Event Type'] == 'recQueue':
                recQueue(future_event_list, state, data, clock, current_customer, cumulative_stat)
            elif current_event['Event Type'] == 'recQueueCar':
                recQueueCar(future_event_list, state, data, clock, cumulative_stat)
            elif current_event['Event Type'] == 'recQueueBus':
                recQueueBus(future_event_list, state, data, clock, cumulative_stat)
            elif current_event['Event Type'] == 'getRec':
                getRec(future_event_list, state, data, clock, current_customer, cumulative_stat)
            elif current_event['Event Type'] == 'foodQueue':
                foodQueue(future_event_list, state, data, clock, current_customer, cumulative_stat)
            elif current_event['Event Type'] == 'getFood':
                getFood(future_event_list, state, data, clock, current_customer, cumulative_stat)
            elif current_event['Event Type'] == 'salQueue':
                salQueue(future_event_list, state, data, clock, current_customer, cumulative_stat)
            elif current_event['Event Type'] == 'endFood':
                endFood(future_event_list, state, data, clock, current_customer, cumulative_stat)
            elif current_event['Event Type'] == 'exitSys':
                exitSys(data, clock, current_customer, cumulative_stat)
            elif current_event['Event Type'] == 'startRecRest':
                startRecRest(future_event_list, state, data, clock)
            elif current_event['Event Type'] == 'startFoodRest':
                startFoodRest(future_event_list, state, data, clock)
            elif current_event['Event Type'] == 'endRecRest':
                endRecRest(future_event_list, state, data, clock, cumulative_stat)
            elif current_event['Event Type'] == 'endFoodRest':
                endFoodRest(future_event_list, state, data, clock, cumulative_stat)

            if LOG_EXCEL:
                output_excel_pandas(future_event_list, state, row_num, cumulative_stat)
                row_num += 1

            future_event_list.remove(current_event)

        else:
            if LOG_EXCEL:
                output_excel_pandas(future_event_list, state, row_num, cumulative_stat)
            future_event_list.clear()

        if state['foodQ'] > maxFoodQ:
            maxFoodQ = state['foodQ']

        if state['salQ'] > maxSalQ:
            maxSalQ = state['salQ']

        if LOG_DATA:
            print("CLOCK", data['EClock'])
            print("CURRENT:", current_event['Event Type'], "CUSTOMER:", current_customer)
            print("FEL:", sorted(future_event_list, key=lambda x: x['Event Time']))
            print("STATE:", state)
            print("CUSTOMERS:", data['Customers'])
            print("CUMULATIVE STATS:", cumulative_stat)
            print("\n")

        if LOG_CHART:
            data['stat_x'].append(data['EClock'])
            data['recQ_stat_y'].append(state['recQ'])
            data['foodQ_stat_y'].append(state['foodQ'])
            data['salQ_stat_y'].append(state['salQ'])

        # Warm up data collection
        if LOG_WARMUP:
            if clock >= timePeriod * WARMUP_STEP:
                if replication_number == 0 and timePeriod == 0:
                    totalPeriod = int(simulation_time / WARMUP_STEP) + 1
                    print(totalPeriod)
                    gData['warmup_y'] = np.zeros(shape=(total_replication, totalPeriod))
                if data['totalCustomers'][1] > 0:
                    gData['warmup_y'][replication_number][timePeriod] = cumulative_stat['recQueue Waiting Time'] \
                                                                        / data['totalCustomers'][1]
                    # (cumulative_stat['recQueue Waiting Time'] + cumulative_stat['recOP Busy Time'])\
                    # / data['totalCustomers'][1]
                else:
                    gData['warmup_y'][replication_number][timePeriod] = 0

                new_row = {
                    'Replication': replication_number + 1,
                    'Period': timePeriod + 1,
                    'Mean Time Spent in Reception': gData['warmup_y'][replication_number][timePeriod]
                }
                warmup_data = warmup_data.append(new_row, ignore_index=True)
                timePeriod += 1

    if LOG_EXCEL:
        excel_formatting_pandas()

    for c in range(1, data['lastCustomer'] + 1):
        tempCustomer = 'C' + str(c)
        for t in range(0, len(data['Customers'][tempCustomer]) - 1):
            # data['totalCustomers'][t] += 1
            gData['times'][replication_number][t] += \
                data['Customers'][tempCustomer][t + 1] - data['Customers'][tempCustomer][t]
            # if t == 0:
            #     cumulative_stat["recQueue Waiting Time"] \
            #         += data['Customers'][tempCustomer][1] - data['Customers'][tempCustomer][0]
            # elif t == 1:
            #     cumulative_stat['recOP Busy Time'] \
            #         += data['Customers'][tempCustomer][2] - data['Customers'][tempCustomer][1]
            # elif t == 3:
            #     cumulative_stat["foodQueue Waiting Time"] \
            #         += data['Customers'][tempCustomer][4] - data['Customers'][tempCustomer][3]
            # elif t == 4:
            #     cumulative_stat['foodOP Busy Time'] \
            #         += data['Customers'][tempCustomer][5] - data['Customers'][tempCustomer][4]
            # elif t == 6:
            #     cumulative_stat["salQueue Waiting Time"] \
            #         += data['Customers'][tempCustomer][7] - data['Customers'][tempCustomer][6]
            # elif t == 7:
            #     cumulative_stat['salOP Busy Time'] \
            #         += data['Customers'][tempCustomer][8] - data['Customers'][tempCustomer][7]
            # elif t == 8:
            #     cumulative_stat['Time in System'] \
            #         += data['Customers'][tempCustomer][9] - data['Customers'][tempCustomer][0]

    for t in range(0, 9):
        gData['times'][replication_number][t] /= data['totalCustomers'][t]

    # Calculating the outputs
    meanLrq = cumulative_stat['recQueue Length'] / simulation_time
    # meanWrq = cumulative_stat['recQueue Waiting Time'] / len(data['Customers'])
    meanBrq = cumulative_stat['recOP Busy Time'] / (simulation_time
                                                    * (STARTING_UNRECOP + STARTING_RECOP) - 4 * TIME_OF_RESTING)

    # meanLfq = cumulative_stat['foodQueue Length'] / simulation_time
    meanWfq = cumulative_stat['foodQueue Waiting Time'] / data['totalCustomers'][4]
    meanBfq = cumulative_stat['foodOP Busy Time'] / (simulation_time
                                                     * (STARTING_UNFOODOP + STARTING_FOODOP) - 4 * TIME_OF_RESTING)

    meanLsq = cumulative_stat['salQueue Length'] / simulation_time
    # meanWsq = cumulative_stat['salQueue Waiting Time'] / len(data['Customers'])
    # meanBsq = cumulative_stat['salOP Busy Time'] / (simulation_time * STARTING_SALOP)

    meanTimeInSystem = cumulative_stat['Time in System'] / data['totalCustomers'][8]

    if LOG_REQ:
        print("Replication Number:", replication_number + 1)
        # Management Request #1
        print("Req1: Mean Time in System=", round(meanTimeInSystem, 3))
        # Management Request #2
        print("Req2: Mean Food Queue Waiting Time=", round(meanWfq, 3))
        # Management Request #3
        print("Req3: Maximum Saloon Queue Length=", maxSalQ)
        print("Req3: Mean Saloon Queue Length=", round(meanLsq, 3))
        # Management Request #4
        print("Req4: Mean Food Operator Busy Time=", round(meanBfq, 3))
        print("Req4: Mean Reception Operator Busy Time=", round(meanBrq, 3))
        # Recommended Request (#5)
        print("Req5: Mean Reception Queue Length=", round(meanLrq, 3))
        # print("Mean Reception Queue Waiting Time=", round(meanWrq, 3))

    new_row = {'Replication': replication_number + 1,
               'R1': round(meanTimeInSystem, 3),
               'R2': round((cumulative_stat['foodQueue Waiting Time'] + cumulative_stat['foodOP Busy Time'])
                           / data['totalCustomers'][4], 3),
               'R3_1': maxSalQ, 'R3_2': round(meanLsq, 3),
               'R4_1': round(meanBfq, 3), 'R4_2': round(meanBrq, 3),
               'R5': round(cumulative_stat['recQueue Waiting Time'] / data['totalCustomers'][0], 3)}
    replication_data = replication_data.append(new_row, ignore_index=True)

    if LOG_CHART:
        if LOG_STEPS:
            print('Drawing queue graph...')

        temp_dataframe = pd.DataFrame(
            {
                "X": data['stat_x'],
                "Reception Queue": data['recQ_stat_y'],
                "Food Queue": data['foodQ_stat_y'],
                "Saloon Queue": data['salQ_stat_y']
            })
        plt.figure(figsize=(simulation_time / 15, 5))
        sb.lineplot(x='X', y='value', hue="variable", data=pd.melt(temp_dataframe, ['X']))
        chartTitle = "Replication:" + str(replication_number + 1)
        plt.title(chartTitle)
        chart_file_name = "Replication#" + str(replication_number + 1) + '-QUEUE.jpg'
        plt.savefig(chart_file_name, bbox_inches='tight', dpi=150)
        plt.show()

        if LOG_STEPS:
            print('Queue graph was drawn successfully')


    if LOG_REQ:
        print("End of Simulation number", replication_number + 1)
    return replication_data, warmup_data


addBus = True
runStat = input("Run simulation with BUS ENTRY? y/n DEF=y")
if runStat == "y":
    addBus = True
elif runStat == "n":
    addBus = False
else:
    addBus = True

excelStat = input("Log the data in Excel? y/n DEF=n")
if excelStat == "y":
    LOG_EXCEL = True
elif excelStat == "n":
    LOG_EXCEL = False
else:
    LOG_EXCEL = False

chartStat = input("Draw charts? y/n DEF=n")
if chartStat == "y":
    LOG_CHART = True
elif chartStat == "n":
    LOG_CHART = False
else:
    LOG_CHART = False

warmupStat = input("Draw warmup charts? y/n DEF=n")
if warmupStat == "y":
    LOG_WARMUP = True
elif warmupStat == "n":
    LOG_WARMUP = False
else:
    LOG_WARMUP = False

replications = int(input("How many replications? "))
gData['times'] = np.zeros(shape=[replications + 1, 9])

if LOG_EXCEL:
    # Maximum length that FEL get (the current event does not count in)
    max_fel = 0
    # Excel file headers
    header_list = ['Step', 'Current Event', 'Clock',
                   'in Rec Queue', 'Busy Rec Operators', 'Idle Rec Operators', 'Time for Rec to Rest?',
                   'in Food Queue', 'Busy Food Operators', 'Idle Food Operators', 'Time for Food to Rest?',
                   'in Saloon Queue', 'Taken Tables',
                   'c(Rec Queue Length)', 'c(Rec Queue Waiting Time)', 'c(Rec Operator Busy Time)',
                   'c(Food Queue Length)', 'c(Food Queue Waiting Time)', 'c(Food Operator Busy Time)',
                   'c(Saloon Queue Length)', 'c(Saloon Queue Waiting Time)', 'c(Table Occupation Time)',
                   'c(Time Spent in System)'
                   ]
    output_df = pd.DataFrame(columns=header_list)

simulationTime = (int(input("Enter the Simulation Time: ")))
if LOG_STEPS:
    print('Starting Simulation...')

if LOG_SENSITIVITY:
    # STARTING_UNRECOP = 2
    # STARTING_SALOP = 15
    TIME_OF_RESTING = 0


for i in range(0, replications):
    if LOG_STEPS:
        print('Running replication #', i + 1)

    replicationData, warmUpData = simulation(simulationTime, i, replicationData, warmUpData, replications)


    if LOG_STEPS:
        print('End of replication #', i + 1)

    # Sesitivity test
    if LOG_SENSITIVITY:
        # STARTING_UNRECOP += 1
        # STARTING_SALOP += 5
        TIME_OF_RESTING += 5

if LOG_SENSITIVITY:
    temp_df = pd.DataFrame({"X": np.arange(0, 31, 5)})
    for (columnName, columnData) in replicationData.iteritems():
        temp_df['Y'] = columnData
        sb.lineplot(x='X', y='value', hue="variable", data=pd.melt(temp_df, ['X']), legend=False)
        plt.title(columnName)

        chartFileName = 'Sensitivity-Test-Req=' + str(columnName) + '.jpg'
        plt.savefig(chartFileName, dpi=300)
        plt.show()

# Calculating replications' Mean and Variance
newRow = {'Replication': 'MEAN:',
          'R1': replicationData['R1'].mean(),
          'R2': replicationData['R2'].mean(),
          'R3_1': replicationData['R3_1'].mean(), 'R3_2': replicationData['R3_2'].mean(),
          'R4_1': replicationData['R4_1'].mean(), 'R4_2': replicationData['R4_2'].mean(),
          'R5': replicationData['R5'].mean()}
replicationData = replicationData.append(newRow, ignore_index=True)

newRow = {'Replication': 'Variance:',
          'R1': replicationData['R1'].var(),
          'R2': replicationData['R2'].var(),
          'R3_1': replicationData['R3_1'].var(), 'R3_2': replicationData['R3_2'].var(),
          'R4_1': replicationData['R4_1'].var(), 'R4_2': replicationData['R4_2'].var(),
          'R5': replicationData['R5'].var()}
replicationData = replicationData.append(newRow, ignore_index=True)

if LOG_STEPS:
    print('Replications data was successfully written to file.')
replicationData.to_excel('replication_output.xlsx', index=False)

if LOG_WARMUP:
    # Write the warmup data to excel
    warmUpData.to_excel('warmup_output.xlsx', index=False)
    if LOG_STEPS:
        print('Warmup data was successfully written to file.')

    # Draw the warmup graph
    if simulationTime % WARMUP_STEP == 0:
        gData['warmup_x'] = np.arange(0, simulationTime + 1, WARMUP_STEP)
    else:
        gData['warmup_x'] = np.arange(0, simulationTime, WARMUP_STEP)
    temp_df = pd.DataFrame({"X": gData['warmup_x']})
    for i in range(0, replications):
        yCount = "R" + str(i + 1)
        temp_df[yCount] = gData['warmup_y'][i]

    if LOG_STEPS:
        print('Drawing warmup graph...')

    plt.figure(figsize=(simulationTime / (WARMUP_STEP * 4), 10))
    sb.lineplot(x='X', y='value', hue="variable", data=pd.melt(temp_df, ['X']), legend=False)
    plt.title("Mean Time in Reception")

    chartFileName = 'Warmup-time=' + str(simulationTime) + '-Bus=' + str(addBus) + '.jpg'
    plt.savefig(chartFileName, dpi=300)
    plt.show()

    if LOG_STEPS:
        print('Warmup graph was drawn successfully.')

for i in range(0, 9):
    for r in range(0, replications):
        gData['times'][replications][i] += gData['times'][r][i]
    gData['times'][replications][i] /= replications

times_header_list = ['T1-T0', 'T2-T1', 'T3-T2', 'T4-T3', 'T5-T4', 'T6-T5', 'T7-T6', 'T8-T7', 'T9-T8']
times = pd.DataFrame(data=gData['times'], columns=times_header_list)
times.to_excel('times_output.xlsx', index=False)

if LOG_STEPS:
    print('Simulation times were successfully written to file.')

print('End of Simulation!')
