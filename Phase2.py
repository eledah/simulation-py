"""
-----------------------------------------------------------------------------------------------
Simulation of One-channel Queueing
Entering dist.~ Poisson with 3 person per hour as mean  # 1/3 hour between two entrances
Servicing dist.~ uniform dist. [10,25] # we should turn them into hours
No limit on Queue's length
People get service in a FIFO system
Outputs : 1- Queue's length Mean
          2- Time waiting in queue Mean
          3-Server efficiency
Starting State = System is empty
-----------------------------------------------------------------------------------------------
"""

import random
import math
from operator import itemgetter
import seaborn as sb
from matplotlib import pyplot as plt
import time as time
# pip install openpyxl
import pandas as pd
import xlsxwriter as xw
from pandas import ExcelWriter
from pandas import ExcelFile

sb.set()
sb.set(style="whitegrid")

stat_x = []
recQ_stat_y = []
foodQ_stat_y = []
salQ_stat_y = []

recOP_stat_y = []
foodOP_stat_y = []


def starting_state():
    # State
    state = dict()
    # state['Server'] = 0
    state['recQueue'] = dict()  # Difference: I use it in End of Service Func to save the queue entering time
    state['foodQueue'] = dict()
    state['salQueue'] = dict()

    state['recQ'] = 0
    state['foodQ'] = 0
    state['salQ'] = 0
    state['recOP'] = 0
    state['unRecOP'] = 5
    state['recRest'] = False
    state['foodOP'] = 0
    state['unFoodOP'] = 2
    state['foodRest'] = False
    state['salOP'] = 30

    # Data Collecting Dict: saves the main 3 times of the customers and time of last event (for calculating Lq, p)
    data_collecting = dict()

    # The event clock
    data_collecting['EClock'] = 0

    # The customer {'Ci':
    # [T(enter Q1), t(left Q1), t(get REC),
    # t(enter Q2), t(left Q2), t(get FOOD),
    # t(enter Q3), t(left Q3), t(end FOOD),
    # t(END)]}
    data_collecting['Customers'] = dict()
    data_collecting['lastCustomer'] = 1

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

    cumulative_stat["Time in System"] = 0

    future_event_list = list()
    # Difference: make an entrance of specific customer (C1)
    FEL_maker(future_event_list, 'recQueue', 0, 'C1')

    # Car and Bus Entrance
    if addBus:
        FEL_maker(future_event_list, 'recQueueBus', 0, '-1')
    FEL_maker(future_event_list, 'recQueueCar', 0, '-1')

    # Rec Rests
    FEL_maker(future_event_list, 'startRecRest', 50, '-1')
    FEL_maker(future_event_list, 'startRecRest', 110, '-1')
    FEL_maker(future_event_list, 'startRecRest', 170, '-1')
    FEL_maker(future_event_list, 'startRecRest', 230, '-1')

    # Food Rests
    FEL_maker(future_event_list, 'startFoodRest', 50, '-1')
    FEL_maker(future_event_list, 'startFoodRest', 110, '-1')
    FEL_maker(future_event_list, 'startFoodRest', 170, '-1')
    FEL_maker(future_event_list, 'startFoodRest', 230, '-1')

    return state, data_collecting, future_event_list, cumulative_stat


def FEL_maker(future_event_list, event_type, clock, customer):
    if event_type == "recQueue":
        event_time = clock + randTime_exp(3)
    elif event_type == "recQueueCar":
        event_time = clock + randTime_exp(5)
    elif event_type == "recQueueBus":
        event_time = clock + randTime_uniform(60, 180)
    elif event_type == "getRec":
        event_time = clock + randTime_tri(1, 2, 4) + randTime_tri(1, 2, 3)
    elif event_type == "startRecRest":
        event_time = clock
    elif event_type == "endRecRest":
        event_time = clock + 10
    elif event_type == "foodQueue":
        event_time = clock + delay('recMove')
    elif event_type == "getFood":
        event_time = clock + randTime_uniform(0.5, 2)
    elif event_type == "startFoodRest":
        event_time = clock
    elif event_type == "endFoodRest":
        event_time = clock + 10
    elif event_type == "salQueue":
        event_time = clock + delay('foodMove')
    elif event_type == "endFood":
        event_time = clock + randTime_tri(10, 20, 30)
    elif event_type == "exitSys":
        event_time = clock + delay('salMove')
    else:
        event_time = 0

    # additional element in event notices (Customer No.)
    new_event = {'Event Type': event_type,
                 'Event Time': round(event_time, 3), 'Customer': customer}
    future_event_list.append(new_event)


def recQueue(future_event_list, state, data, clock, customer, cum_stat, create_next=True):
    data['Customers'][customer] = []  # Add a place for the new customer

    cum_stat['recQueue Length'] += state['recQ'] * (clock - data['EClock'])

    # Is the server busy?
    # NO
    if state['unRecOP'] > 0:
        data['Customers'][customer] = [clock, clock]  # T(ent) = T(left q) = clock
        state['recOP'] = state['recOP'] + 1  # Make 1 Operator Busy
        state['unRecOP'] = state['unRecOP'] - 1  # Remove 1 Free Operator

        FEL_maker(future_event_list, 'getRec', clock, customer)  # Say when does this customer's service end?

    # YES
    else:
        data['Customers'][customer] = [clock]  # T(ent) = clock
        state['recQ'] = state['recQ'] + 1
        state['recQueue'][customer] = clock

    # put the current clock on the last event clock  for the next event (what a sentence :)) )
    advanceTime(data, clock)

    if create_next:
        # Extracting the customer num
        data['lastCustomer'] += 1
        FEL_maker(future_event_list, 'recQueue', clock, 'C' + str(data['lastCustomer']))


def carPassengers():
    rand = random.random()
    if 0.2 > rand > 0:
        return 1
    elif 0.5 > rand > 0.2:
        return 2
    elif 0.8 > rand > 0.5:
        return 3
    return 4


def recQueueCar(future_event_list, state, data, clock, customer, cum_stat):
    passengers = carPassengers()
    for i in range(0, passengers):
        data['lastCustomer'] += 1
        # print('C' + str(data['lastCustomer']))
        recQueue(future_event_list, state, data, clock, 'C' + str(data['lastCustomer']), cum_stat, False)
    advanceTime(data, clock)
    FEL_maker(future_event_list, 'recQueueCar', clock, '-1')


def recQueueBus(future_event_list, state, data, clock, customer, cum_stat):
    passengers = randTime_poisson(30)
    for i in range(0, passengers):
        data['lastCustomer'] += 1
        # print('C' + str(data['lastCustomer']))
        recQueue(future_event_list, state, data, clock, 'C' + str(data['lastCustomer']), cum_stat, False)
    advanceTime(data, clock)


def getRec(future_event_list, state, data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)

    # Send the customer off
    FEL_maker(future_event_list, "foodQueue", clock, customer)

    # Make the operator idle
    state['recOP'] = state['recOP'] - 1

    if state['recRest']:
        # It's time for resting
        FEL_maker(future_event_list, "endRecRest", clock, '-1')
        state['recRest'] = False
    else:
        # It's not time for resting
        state['unRecOP'] = state['unRecOP'] + 1
        if state['recQ'] > 0:
            # There's someone in the queue. Make the operator busy
            state['recQ'] = state['recQ'] - 1
            state['recOP'] = state['recOP'] + 1
            state['unRecOP'] = state['unRecOP'] - 1

            # Send in the next customer
            firstCustomer = firstInRecQueue(state)
            data['Customers'][firstCustomer].append(clock)
            FEL_maker(future_event_list, "getRec", clock, firstCustomer)

            # Delete the customer from queue
            del state['recQueue'][firstCustomer]

            cum_stat["recQueue Waiting Time"] += (
                    data['Customers'][firstCustomer][1] - data['Customers'][firstCustomer][0])
    advanceTime(data, clock)


def foodQueue(future_event_list, state, data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)

    cum_stat['foodQueue Length'] += state['foodQ'] * (clock - data['EClock'])

    cum_stat['recOP Busy Time'] += data['Customers'][customer][2] - data['Customers'][customer][1]

    # Is the server busy?
    # NO
    if state['unFoodOP'] > 0:
        # T(ent) = T(left q) = clock
        data['Customers'][customer].append(clock)
        data['Customers'][customer].append(clock)
        # Make 1 Operator Busy
        state['foodOP'] = state['foodOP'] + 1
        # Remove 1 Free Operator
        state['unFoodOP'] = state['unFoodOP'] - 1

        FEL_maker(future_event_list, 'getFood', clock, customer)  # Say when does this customer's service end?

    # YES
    else:
        data['Customers'][customer].append(clock)  # T(ent) = clock
        state['foodQ'] = state['foodQ'] + 1
        state['foodQueue'][customer] = clock  # state['Queue'] += 1

    advanceTime(data, clock)


def getFood(future_event_list, state, data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)

    # Send the customer off
    FEL_maker(future_event_list, "salQueue", clock, customer)

    # Make the operator idle
    state['foodOP'] = state['foodOP'] - 1

    if state['foodRest']:
        # It's time for resting
        FEL_maker(future_event_list, "endFoodRest", clock, '-1')
        state['foodRest'] = False
    else:
        # It's not time for resting
        state['unFoodOP'] = state['unFoodOP'] + 1
        if state['foodQ'] > 0:
            # There's someone in the queue. Make the operator busy
            state['foodQ'] = state['foodQ'] - 1
            state['foodOP'] = state['foodOP'] + 1
            state['unFoodOP'] = state['unFoodOP'] - 1

            # Send in the next customer
            firstCustomer = firstInFoodQueue(state)
            data['Customers'][firstCustomer].append(clock)
            FEL_maker(future_event_list, "getFood", clock, firstCustomer)

            # Delete the customer from queue
            del state['foodQueue'][firstCustomer]

    advanceTime(data, clock)


def salQueue(future_event_list, state, data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)

    cum_stat['salQueue Length'] += state['salQ'] * (clock - data['EClock'])

    cum_stat['foodOP Busy Time'] += data['Customers'][customer][5] - data['Customers'][customer][4]
    cum_stat["foodQueue Waiting Time"] += data['Customers'][customer][4] - data['Customers'][customer][3]
    # Are the tables full
    # NO
    if state['salOP'] > 0:
        # T(ent) = T(left q) = clock
        data['Customers'][customer].append(clock)
        data['Customers'][customer].append(clock)
        # Make 1 Table Busy
        state['salOP'] = state['salOP'] + 1

        FEL_maker(future_event_list, 'endFood', clock, customer)  # Say when does this customer's service end?

    # YES
    else:
        data['Customers'][customer].append(clock)  # T(ent) = clock
        state['salQ'] = state['salQ'] + 1
        state['salQueue'][customer] = clock  # state['Queue'] += 1

    advanceTime(data, clock)


def endFood(future_event_list, state, data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)

    cum_stat['recOP Busy Time'] += data['Customers'][customer][8] - data['Customers'][customer][7]
    # Send the customer off
    FEL_maker(future_event_list, "exitSys", clock, customer)

    # Set the table free
    state['salOP'] = state['salOP'] - 1

    if state['salQ'] > 0:
        # There's someone in the queue. Make the operator busy
        state['salQ'] = state['salQ'] - 1
        state['salOP'] = state['salOP'] + 1

        # Send in the next customer
        firstCustomer = firstInSalQueue(state)
        data['Customers'][firstCustomer].append(clock)
        FEL_maker(future_event_list, "endFood", clock, firstCustomer)

        # Delete the customer from queue
        del state['salQueue'][firstCustomer]

        cum_stat["salQueue Waiting Time"] += (
                data['Customers'][firstCustomer][7] - data['Customers'][firstCustomer][6])
    advanceTime(data, clock)


def exitSys(future_event_list, state, data, clock, customer, cum_stat):
    cum_stat['Time in System'] += data['Customers'][customer][9] - data['Customers'][customer][0]
    advanceTime(data, clock)


def startRecRest(future_event_list, state, data, clock, cum_stat):
    if state['unRecOP'] > 0:
        state['unRecOP'] = state['unRecOP'] - 1
        FEL_maker(future_event_list, 'endRecRest', clock, '-1')
    else:
        state['recRest'] = True
    advanceTime(data, clock)


def startFoodRest(future_event_list, state, data, clock, cum_stat):
    if state['unFoodOP'] > 0:
        state['unFoodOP'] = state['unFoodOP'] - 1
        FEL_maker(future_event_list, 'endFoodRest', clock, '-1')
    else:
        state['foodRest'] = True
    advanceTime(data, clock)


def endRecRest(future_event_list, state, data, clock, cum_stat):
    if state['recQ'] > 0:
        state['recQ'] = state['recQ'] - 1
        state['recOP'] = state['recOP'] + 1

        firstCustomer = firstInRecQueue(state)
        FEL_maker(future_event_list, 'getRec', clock, firstCustomer)
        del state['recQueue'][firstCustomer]
    else:
        state['unRecOP'] = state['unRecOP'] + 1
    advanceTime(data, clock)


def endFoodRest(future_event_list, state, data, clock, cum_stat):
    if state['foodQ'] > 0:
        state['foodQ'] = state['foodQ'] - 1
        state['foodOP'] = state['foodOP'] + 1

        firstCustomer = firstInFoodQueue(state)
        FEL_maker(future_event_list, 'getFood', clock, firstCustomer)
        del state['foodQueue'][firstCustomer]
    else:
        state['unFoodOP'] = state['unFoodOP'] + 1
    advanceTime(data, clock)


def firstInRecQueue(state):
    firstCustomer = min(state['recQueue'], key=lambda k: state['recQueue'][k])
    return firstCustomer


def firstInFoodQueue(state):
    firstCustomer = min(state['foodQueue'], key=lambda k: state['foodQueue'][k])
    return firstCustomer


def firstInSalQueue(state):
    firstCustomer = min(state['salQueue'], key=lambda k: state['salQueue'][k])
    return firstCustomer


# uniform dist
def randTime_uniform(a, b):
    return (b - a) * random.random() + a


# exp dist
def randTime_exp(mean):
    return math.log(random.random()) / (-1 / mean)


# Triangular dist
def randTime_tri(a, b, c):
    # a->start b->end c->mode
    F_C = (c - a) / (b - a)
    newRandom = random.random()
    if F_C > newRandom > 0:
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
    if name == "recMove" or name == "recFood":
        return randTime_exp(0.5)
    elif name == "salMove":
        return randTime_exp(1)
    return 0


def advanceTime(data, clock):
    data['EClock'] = clock


def output_excel_pandas(future_event_list, state, row_num, cumulative_stat):
    global output_df  # Use the dataframe global variable
    global max_fel  # Use the max_fel global variable
    future_event_list = sorted(future_event_list
                               , key=lambda x: x['Event Time'])  # I add the current event so I don't need this
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
    worksheet.set_column(14, 23, 27 )
    worksheet.set_column(5, 4 + 2 * max_fel, 19)
    cell_format = workbook.add_format()
    cell_format.set_align('center')
    for row in range(len(output_df)):
        worksheet.set_row(row + 1, None, cell_format)

    # Save the excel
    excel_output.save()


def simulation(simulation_time):
    maxFoodQ = 0
    row_num = 1
    state, data, future_event_list, cumulative_stat = starting_state()  # Starting state
    clock = 0
    future_event_list.append({'Event Type': 'End of Simulation', 'Event Time': simulation_time})  # Add specific events

    # Continue till the simulation time ends
    while clock < simulation_time:  # while current_event['Event Type'] != 'End of Simulation'

        sorted_fel = sorted(future_event_list, key=lambda x: x['Event Time'])  # Sort the FEL based on event times
        # sorted_fel = sorted(future_event_list, key=itemgetter(1))                 # Another way to sort
        current_event = sorted_fel[0]  # The first element is what happening now
        clock = current_event['Event Time']  # Move the time forward
        # print(current_event['Event Type'])

        if clock < simulation_time:
            current_customer = current_event['Customer']
            if current_event['Event Type'] == 'recQueue':
                recQueue(future_event_list, state, data, clock, current_customer, cumulative_stat)
            elif current_event['Event Type'] == 'recQueueCar':
                recQueueCar(future_event_list, state, data, clock, current_customer, cumulative_stat)
            elif current_event['Event Type'] == 'recQueueBus':
                recQueueBus(future_event_list, state, data, clock, current_customer, cumulative_stat)
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
                exitSys(future_event_list, state, data, clock, current_customer, cumulative_stat)
            elif current_event['Event Type'] == 'startRecRest':
                startRecRest(future_event_list, state, data, clock, cumulative_stat)
            elif current_event['Event Type'] == 'startFoodRest':
                startFoodRest(future_event_list, state, data, clock, cumulative_stat)
            elif current_event['Event Type'] == 'endRecRest':
                endRecRest(future_event_list, state, data, clock, cumulative_stat)
            elif current_event['Event Type'] == 'endFoodRest':
                endFoodRest(future_event_list, state, data, clock, cumulative_stat)

            output_excel_pandas(future_event_list, state, row_num, cumulative_stat)
            row_num += 1
            future_event_list.remove(current_event)

            if state['foodQ'] > maxFoodQ:
                maxFoodQ = state['foodQ']

            print("CLOCK", data['EClock'])
            print("CURRENT:", current_event['Event Type'], "CUSTOMER:", current_customer)
            print("FEL:", sorted(future_event_list, key=lambda x: x['Event Time']))
            print("STATE:", state)
            print("CUSTOMERS:", data['Customers'])
            print("CUMULATIVE STATS:", cumulative_stat)

            print("\n")

            # print("foodQ:", state['foodQ'], "len(foodQueue):", len(state['foodQueue']))
            # if state['foodQ'] != len(state['foodQueue']):
            #     time.sleep(3)

            stat_x.append(data['EClock'])

            recQ_stat_y.append(state['recQ'])
            foodQ_stat_y.append(state['foodQ'])

        # Do the corrections
        else:
            output_excel_pandas(future_event_list, state, row_num, cumulative_stat)
            future_event_list.clear()
            print("CLOCK", clock)
            print("CURRENT:", current_event['Event Type'])
            print("FEL:", sorted(future_event_list, key=lambda x: x['Event Time']))
            print("STATE:", state)
            print("CUSTOMERS:", data['Customers'])
            print("CUMULATIVE STATS:", cumulative_stat)

            print("\n")

    excel_formatting_pandas()
    # Calculating the outputs
    meanLrq = cumulative_stat['recQueue Length'] / simulation_time
    meanWrq = cumulative_stat['recQueue Waiting Time'] / len(data['Customers'])
    meanBrq = cumulative_stat['recOP Busy Time'] / (simulation_time * 5)

    meanLfq = cumulative_stat['foodQueue Length'] / simulation_time
    meanWfq = cumulative_stat['foodQueue Waiting Time'] / len(data['Customers'])
    meanBfq = cumulative_stat['foodOP Busy Time'] / (simulation_time * 2)

    meanTimeInSystem = cumulative_stat['Time in System'] / len(data['Customers'])

    # Management Request #1
    print("Req1: Mean Time in System=", round(meanTimeInSystem, 3))
    # Management Request #2
    print("Req2: Mean Food Queue Waiting Time=", round(meanWfq, 3))
    # Management Request #3
    print("Req3: Maximum Food Queue Length=", maxFoodQ)
    print("Req3: Mean Food Queue Length=", round(meanLfq, 3))
    # Management Request #4
    print("Req4: Mean Food Operator Busy Time=", round(meanBfq, 3))
    print("Req4: Mean Reception Operator Busy Time=", round(meanBrq, 3))

    print("Mean Reception Queue Length=", round(meanLrq, 3))
    print("Mean Reception Queue Waiting Time=", round(meanWrq, 3))

    temp_df = pd.DataFrame(
        {
            "X": stat_x,
            "recQ": recQ_stat_y,
            "foodQ": foodQ_stat_y
        })
    plt.figure(figsize=(simulation_time / 10, 10))
    sb.lineplot(x='X', y='value', hue="variable", data=pd.melt(temp_df, ['X']))
    plt.show()

    print("End of Simulation.")


addBus = True
runStat = input("Run simulation without BUS ENTRY? y/n")
if runStat == "y":
    addBus = False

# new_row = [row_num, future_event_list[0]['Event Type'], future_event_list[0]['Event Time'],  # Step, Current, Clock
#            state['recQ'], state['recOP'], state['unRecOP'], state['recRest'],  # Reception
#            state['foodQ'], state['foodOP'], state['unFoodOP'], state['foodRest'],  # Food
#            state['salQ'], state['salOP'],  # Saloon
#            cumulative_stat["recQueue Length"],
#            cumulative_stat["recQueue Waiting Time"],
#            cumulative_stat["recOP Busy Time"],
#
#            cumulative_stat["foodQueue Length"],
#            cumulative_stat["foodQueue Waiting Time"],
#            cumulative_stat["foodOP Busy Time"],
#
#            cumulative_stat["salQueue Length"],
#            cumulative_stat["salQueue Waiting Time"],
#            cumulative_stat["salOP Busy Time"],
#
#            cumulative_stat["Time in System"]
#            ]
max_fel = 0  # Maximum length that FEL get (the current event does not count in)
header_list = ['Step', 'Current Event', 'Clock',
               'in Rec Queue', 'Busy Rec Operators', 'Idle Rec Operators', 'Time for Rec to Rest?',
               'in Food Queue', 'Busy Food Operators', 'Idle Food Operators', 'Time for Food to Rest?',
               'in Saloon Queue', 'Taken Tables',
               'c(Rec Queue Length)', 'c(Rec Queue Waiting Time)', 'c(Rec Operator Busy Time)',
               'c(Food Queue Length)', 'c(Food Queue Waiting Time)', 'c(Food Operator Busy Time)',
               'c(Saloon Queue Length)', 'c(Saloon Queue Waiting Time)', 'c(Table Occupation Time)',
               'c(Time Spent in System)'
               ]  # Headers at First
output_df = pd.DataFrame(columns=header_list)  # Create a dataframe with the predefined headers

simulation(int(input("Enter the Simulation Time: ")))
