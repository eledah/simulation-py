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


def starting_state():
    # State
    state = dict()
    # state['Server'] = 0
    state['recQueue'] = dict()  # Difference: I use it in End of Service Func to save the queue entering time
    state['foodQueue'] = dict()

    state['recQ'] = 0
    state['foodQ'] = 0
    state['salQ'] = 0
    state['recOP'] = 0
    state['unRecOP'] = 5
    state['recRest'] = False
    state['foodOP'] = 0
    state['unFoodOP'] = 5
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

    # Cumulative statistics
    cumulative_stat = dict()
    cumulative_stat["recQueue Length"] = 0  # for Lq
    cumulative_stat["recQueue Waiting Time"] = 0  # for Wq
    cumulative_stat["recOP Busy Time"] = 0  # for p

    cumulative_stat["foodQueue Length"] = 0  # for Lq
    cumulative_stat["foodQueue Waiting Time"] = 0  # for Wq
    cumulative_stat["foodOP Busy Time"] = 0  # for p

    future_event_list = list()
    FEL_maker(future_event_list, 'recQueue', 0, 'C1')  # Difference: make an entrance of specific customer (C1)
    FEL_maker(future_event_list, 'startRecRest', 5, '-1')
    FEL_maker(future_event_list, 'startFoodRest', 5, '-1')
    return state, data_collecting, future_event_list, cumulative_stat


def FEL_maker(future_event_list, event_type, clock, customer):
    rand_num = random.random()

    if event_type == "recQueue":
        event_time = clock + round((-1 / 10) * math.log(rand_num), 3)
    elif event_type == "getRec":
        event_time = clock + round((1 / 2) + (1 / 4) * rand_num, 3)
    elif event_type == "startRecRest":
        event_time = clock
    elif event_type == "endRecRest":
        event_time = clock + 2
    elif event_type == "foodQueue":
        event_time = clock + 0.5
    elif event_type == "getFood":
        event_time = clock + round((1 / 2) + (1 / 4) * rand_num, 3)
    elif event_type == "startFoodRest":
        event_time = clock
    elif event_type == "endFoodRest":
        event_time = clock + 0.5
    else:
        event_time = clock + round((-1 / 10) * math.log(rand_num), 3)

    new_event = {'Event Type': event_type,
                 'Event Time': event_time, 'Customer': customer}  # additional element in event notices (Customer No.)
    future_event_list.append(new_event)


def recQueue(future_event_list, state, data, clock, customer, cum_stat):
    # print("FEL:", future_event_list, "\nState:", state, "\nclock:", clock, "\ncustomer:", customer,
    #       "\n\n")
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
        cum_stat['recOP Busy Time'] += (clock - data['EClock'])  # Server is busy just before this time
        data['Customers'][customer] = [clock]  # T(ent) = clock
        state['recQ'] = state['recQ'] + 1
        state['recQueue'][customer] = clock  # state['recQueue'] += 1

    # put the current clock on the last event clock  for the next event (what a sentence :)) )
    data['EClock'] = clock

    # Extracting the customer num
    customer_num = int(customer[1:])
    customer_num += 1
    FEL_maker(future_event_list, 'recQueue', clock, 'C' + str(customer_num))  # predict the next customer's Arrival


def recQueueCar(future_event_list, state, data, clock, customer, cum_stat):
    pass


def recQueueBus(future_event_list, state, data, clock, customer, cum_stat):
    pass


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


def foodQueue(future_event_list, state, data, clock, customer, cum_stat):
    data['Customers'][customer].append(clock)

    cum_stat['foodQueue Length'] += state['foodQ'] * (clock - data['EClock'])

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
        cum_stat['foodOP Busy Time'] += (clock - data['EClock'])  # Server is busy just before this time
        data['Customers'][customer].append(clock)  # T(ent) = clock
        state['foodQ'] = state['foodQ'] + 1
        state['foodQueue'][customer] = clock  # state['Queue'] += 1

    # put the current clock on the last event clock  for the next event (what a sentence :)) )
    data['EClock'] = clock


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

            cum_stat["foodQueue Waiting Time"] += (
                    data['Customers'][firstCustomer][4] - data['Customers'][firstCustomer][3])


def salQueue(future_event_list, state, data, clock, customer, cum_stat):
    pass


def endFood(future_event_list, state, data, clock, customer, cum_stat):
    pass


def exitSys(future_event_list, state, data, clock, customer, cum_stat):
    pass


def startRecRest(future_event_list, state, data, clock, cum_stat):
    if state['unRecOP'] > 0:
        state['unRecOP'] = state['unRecOP'] - 1
        FEL_maker(future_event_list, 'endRecRest', clock, '-1')
    else:
        state['recRest'] = True


def startFoodRest(future_event_list, state, data, clock, cum_stat):
    pass


def endRecRest(future_event_list, state, data, clock, cum_stat):
    if state['recQ'] > 0:
        state['recQ'] = state['recQ'] - 1
        state['recOP'] = state['recOP'] + 1

        firstCustomer = firstInRecQueue(state)
        FEL_maker(future_event_list, 'getRec', clock, firstCustomer)
    else:
        state['unRecOP'] = state['unRecOP'] + 1


def endFoodRest(future_event_list, state, data, clock, cum_stat):
    pass


def firstInRecQueue(state):
    firstCustomer = min(state['recQueue'], key=lambda k: state['recQueue'][k])
    return firstCustomer


def firstInFoodQueue(state):
    firstCustomer = min(state['foodQueue'], key=lambda k: state['foodQueue'][k])
    return firstCustomer


def simulation(simulation_time):
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
            future_event_list.remove(current_event)

            print(state)
            print(data['Customers'])
            print(data['EClock'])
            print(cumulative_stat)
            print("\n")

        # If the simulation time is over empty the FEL and all the customers (As we debate in class it is wrong)
        # Do the corrections
        else:
            future_event_list = []
            # keys = [k for k in state['Queue']]
            # for key in keys:
            #     del data['Customers'][key]
    #
    # Calculating the outputs
    queue_len = cumulative_stat['recQueue Length'] / simulation_time
    queue_waiting_time = cumulative_stat['recQueue Waiting Time'] / len(data['Customers'])
    server_util = cumulative_stat['recOP Busy Time'] / simulation_time
    print("Lq = " + str(queue_len) + "\nWq = " + str(queue_waiting_time) + "\np = " + str(server_util))
    print("Little's Law is almost correct : " + str(queue_len / queue_waiting_time))  # Checking the little's law

    print("End!")


simulation(int(input("Enter the Simulation Time: ")))
