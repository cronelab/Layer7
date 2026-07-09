#! /bin/env python
#
# Michael Gibson 17 July 2015
# Modified Adrian Foy February 2021

import logging, sys, struct, math, os, time
import numpy as np
import time

from pathlib import Path

from intanutil.read_header import read_header
from intanutil.get_bytes_per_data_block import get_bytes_per_data_block
from intanutil.read_one_data_block import read_one_data_block
from intanutil.notch_filter import notch_filter
from intanutil.data_to_result import data_to_result


AMPLIFIER_TO_MICROVOLTS = 0.195
AUXILIARY_TO_VOLTS = 37.4e-6
SUPPLY_TO_VOLTS = 74.8e-6
TEMP_TO_C = 0.01

BOARD_MODE_ADC_TO_VOLTS = {
    1: 152.59e-6,
    13: 312.5e-6,
}
BOARD_MODE_ADC_TO_VOLTS_DEFAULT = 50.354e-6


def read_rhd_data(header, fid, filesize):
    """ Parse data following the RHD header.
    """
    data = {}

    # Determine how many samples the data file contains.
    bytes_per_block = get_bytes_per_data_block(header)

    # How many data blocks remain in this file?
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    if bytes_remaining % bytes_per_block != 0:
        logging.error("Expected bytes per block: %s", bytes_per_block)
        logging.error("Excess bytes: %s", bytes_remaining % bytes_per_block)
        raise Exception('Something is wrong with file size : should have a whole number of data blocks')

    num_data_blocks = int(bytes_remaining / bytes_per_block)

    num_amplifier_samples = header['num_samples_per_data_block'] * num_data_blocks
    num_aux_input_samples = int((header['num_samples_per_data_block'] / 4) * num_data_blocks)
    num_supply_voltage_samples = 1 * num_data_blocks
    num_board_adc_samples = header['num_samples_per_data_block'] * num_data_blocks
    num_board_dig_in_samples = header['num_samples_per_data_block'] * num_data_blocks
    num_board_dig_out_samples = header['num_samples_per_data_block'] * num_data_blocks

    record_time = num_amplifier_samples / header['sample_rate']

    if num_data_blocks:
        logging.info('File contains {:0.3f} seconds of data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(record_time, header['sample_rate'] / 1000))
    else:
        logging.info('Header file contains no data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(header['sample_rate'] / 1000))

    if data_present:
        # Pre-allocate memory for data.
        logging.info('')
        logging.info('Allocating memory for data...')

        if (header['version']['major'] == 1 and header['version']['minor'] >= 2) or (header['version']['major'] > 1):
            data['t_amplifier'] = np.zeros(num_amplifier_samples, dtype=int)
        else:
            data['t_amplifier'] = np.zeros(num_amplifier_samples, dtype=np.uint)

        data['amplifier_data'] = np.zeros([header['num_amplifier_channels'], num_amplifier_samples], dtype=np.double)
        data['aux_input_data'] = np.zeros([header['num_aux_input_channels'], num_aux_input_samples], dtype=np.double)
        data['supply_voltage_data'] = np.zeros([header['num_supply_voltage_channels'], num_supply_voltage_samples], dtype=np.uint)
        data['temp_sensor_data'] = np.zeros([header['num_temp_sensor_channels'], num_supply_voltage_samples], dtype=np.uint)
        data['board_adc_data'] = np.zeros([header['num_board_adc_channels'], num_board_adc_samples], dtype=np.uint)
        
        # by default, this script interprets digital events (digital inputs and outputs) as booleans
        # if unsigned int values are preferred(0 for False, 1 for True), replace the 'dtype=np.bool' argument with 'dtype=np.uint' as shown
        # the commented line below illustrates this for digital input data; the same can be done for digital out
        
        #data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=np.uint)
        data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=bool)
        data['board_dig_in_raw'] = np.zeros(num_board_dig_in_samples, dtype=np.uint)
        
        data['board_dig_out_data'] = np.zeros([header['num_board_dig_out_channels'], num_board_dig_out_samples], dtype=bool)
        data['board_dig_out_raw'] = np.zeros(num_board_dig_out_samples, dtype=np.uint)

        # Read sampled data from file.
        logging.info('Reading data from file...')

        # Initialize indices used in looping
        indices = {}
        indices['amplifier'] = 0
        indices['aux_input'] = 0
        indices['supply_voltage'] = 0
        indices['board_adc'] = 0
        indices['board_dig_in'] = 0
        indices['board_dig_out'] = 0

        increment = 10
        percent_done = increment
        for i in range(num_data_blocks):
            read_one_data_block(data, header, indices, fid)

            # Increment indices
            indices['amplifier'] += header['num_samples_per_data_block']
            indices['aux_input'] += int(header['num_samples_per_data_block'] / 4)
            indices['supply_voltage'] += 1
            indices['board_adc'] += header['num_samples_per_data_block']
            indices['board_dig_in'] += header['num_samples_per_data_block']
            indices['board_dig_out'] += header['num_samples_per_data_block']            

            fraction_done = 100 * (1.0 * i / num_data_blocks)
            if fraction_done >= percent_done:
                logging.info('{}% done...'.format(percent_done))
                percent_done = percent_done + increment

        # Make sure we have read exactly the right amount of data.
        bytes_remaining = filesize - fid.tell()
        if bytes_remaining != 0: raise Exception('Error: End of file not reached.')


    # Close data file.
    fid.close()

    return data

def read_dat_data(header, path, rescale_to_voltage =1):
    """ Parse data from DAT files at the given path.
    """

    data = {}

    root_path = Path(path).parent

    with open(root_path / 'time.dat', 'rb') as ifs:
        entry = 't_amplifier'
        array = np.fromfile(ifs, dtype='<i4')

        data[entry] = array

    num_samples = len(data['t_amplifier'])

    num_amplifier_samples = num_samples
    num_aux_input_samples = num_samples >> 2
    num_supply_voltage_samples = int(num_samples / header['num_samples_per_data_block'])
    num_board_adc_samples = num_samples
    num_board_dig_in_samples = num_samples
    num_board_dig_out_samples = num_samples

    logging.info('Header file contains no data. Flagged as header for DAT streams. Amplifiers were sampled at {:0.2f} kS/s.'.format(header['sample_rate'] / 1000))

    # Pre-allocate memory for data.
    logging.info('')
    logging.info('Allocating memory for data...')

    data['amplifier_data'] = np.zeros([header['num_amplifier_channels'], num_amplifier_samples], dtype=np.double)
    data['aux_input_data'] = np.zeros([header['num_aux_input_channels'], num_aux_input_samples], dtype=np.double)
    data['supply_voltage_data'] = np.zeros([header['num_supply_voltage_channels'], num_supply_voltage_samples], dtype=np.uint)
    data['temp_sensor_data'] = np.zeros([header['num_temp_sensor_channels'], num_supply_voltage_samples], dtype=np.uint)
    data['board_adc_data'] = np.zeros([header['num_board_adc_channels'], num_board_adc_samples], dtype=np.uint)
    data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=bool)
    data['board_dig_in_raw'] = np.zeros(num_board_dig_in_samples, dtype=np.uint)
    
    data['board_dig_out_data'] = np.zeros([header['num_board_dig_out_channels'], num_board_dig_out_samples], dtype=bool)
    data['board_dig_out_raw'] = np.zeros(num_board_dig_out_samples, dtype=np.uint)

    # Read sampled data from file.
    logging.info('Reading data from file...')

    # Note the Fortran ordering specified throughout. This has to do with
    # the difference between the C-style ordering assumed by the above
    # matrices and the implicit Fortran-style ordering of the DAT files.
    #
    # A nice side effect of this is that persing and processing can be
    # done much more quickly, as the memory ordering aligns with the
    # temporal nature of the data.
    #
    # If conversion back to C-style ordering is necessary, use the
    # numpy.ascontiguousarray method to rearrange the underlying array.

    def fn(filename, entry, f=None, dtype="<u2"):
        # default: unsigned 2-bit integer
        with open(root_path / filename, 'rb') as ifs:
            # array = np.fromfile(ifs, dtype='<u2')
            array = np.fromfile(ifs, dtype=dtype)

        # Apply array filter, if provided.
        array = f(array) if f else array

        # Reshape the array, also converting C-style ordering to Fortran-style ordering.
        # data[entry] = np.ascontiguousarray(array.reshape(data[entry].shape, order='F')).
        # data[entry] = np.ascontiguousarray(array.reshape(data[entry].shape, order='C'))

        # data[entry] = array.reshape(data[entry].shape, order='F')
        test_for_synthetic_data = 0
        if test_for_synthetic_data:
        # if entry == 'amplifier_data':
            data[entry] = array.reshape([1024, 1198080], order='F')
        else:
            data[entry] = array.reshape(data[entry].shape, order='F')

    if header['num_amplifier_channels']:
        # if rescale_to_voltage == 0:
        print('Reading the amplifier data as an signed integer array ..')
        fn('amplifier.dat', 'amplifier_data', dtype = "<i2")
            # pass
        # else:
            # fn('amplifier.dat', 'amplifier_data', lambda x: x + 2**15)
                

    if header['num_aux_input_channels']:
        fn('auxiliary.dat', 'aux_input_data', lambda x: x[::4])

    if header['num_supply_voltage_channels']:
        fn('supply.dat', 'supply_voltage_data', lambda x: x[::128])

    if header['num_board_adc_channels']:
        try:
            # fn('analogin.dat', 'board_adc_data', dtype = "<i2")       # signed
            fn('analogin.dat', 'board_adc_data')                        # unsigned
        except:
            print("ISSUE WITH ANALOGIN")
    if header['num_board_dig_in_channels']:
        fn('digitalin.dat', 'board_dig_in_raw')

    if header['num_board_dig_out_channels']:
        fn('digitalout.dat', 'board_dig_out_raw')

    return data


def read_dat_data_digital_only(header, path):
    """ Parse data from DAT files at the given path.
    """

    data = {}

    root_path = Path(path).parent

    with open(root_path / 'time.dat', 'rb') as ifs:
        entry = 't_amplifier'
        array = np.fromfile(ifs, dtype='<i4')

        data[entry] = array

    num_samples = len(data['t_amplifier'])

    num_amplifier_samples = num_samples
    num_aux_input_samples = num_samples >> 2
    num_supply_voltage_samples = int(num_samples / header['num_samples_per_data_block'])
    num_board_adc_samples = num_samples
    num_board_dig_in_samples = num_samples
    num_board_dig_out_samples = num_samples

    logging.info('Header file contains no data. Flagged as header for DAT streams. Amplifiers were sampled at {:0.2f} kS/s.'.format(header['sample_rate'] / 1000))

    # Pre-allocate memory for data.
    logging.info('')
    logging.info('Allocating memory for data...')

    data['amplifier_data'] = np.zeros([header['num_amplifier_channels'], num_amplifier_samples], dtype=np.double)
    data['aux_input_data'] = np.zeros([header['num_aux_input_channels'], num_aux_input_samples], dtype=np.double)
    data['supply_voltage_data'] = np.zeros([header['num_supply_voltage_channels'], num_supply_voltage_samples], dtype=np.uint)
    data['temp_sensor_data'] = np.zeros([header['num_temp_sensor_channels'], num_supply_voltage_samples], dtype=np.uint)
    data['board_adc_data'] = np.zeros([header['num_board_adc_channels'], num_board_adc_samples], dtype=np.uint)
    data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=bool)
    data['board_dig_in_raw'] = np.zeros(num_board_dig_in_samples, dtype=np.uint)
    
    data['board_dig_out_data'] = np.zeros([header['num_board_dig_out_channels'], num_board_dig_out_samples], dtype=bool)
    data['board_dig_out_raw'] = np.zeros(num_board_dig_out_samples, dtype=np.uint)

    # Read sampled data from file.
    logging.info('Reading data from file...')

    # Note the Fortran ordering specified throughout. This has to do with
    # the difference between the C-style ordering assumed by the above
    # matrices and the implicit Fortran-style ordering of the DAT files.
    #
    # A nice side effect of this is that persing and processing can be
    # done much more quickly, as the memory ordering aligns with the
    # temporal nature of the data.
    #
    # If conversion back to C-style ordering is necessary, use the
    # numpy.ascontiguousarray method to rearrange the underlying array.

    def fn(filename, entry, f=None):
        with open(root_path / filename, 'rb') as ifs:
            array = np.fromfile(ifs, dtype='<u2')

        # Apply array filter, if provided.
        array = f(array) if f else array

        # Reshape the array, also converting Fortran-style ordering to C-style ordering.
        # data[entry] = np.ascontiguousarray(array.reshape(data[entry].shape, order='F')).
        data[entry] = np.ascontiguousarray(array.reshape(data[entry].shape, order='C'))
        # data[entry] = array.reshape(data[entry].shape, order='F')

    # if header['num_amplifier_channels']:
    #     fn('amplifier.dat', 'amplifier_data', lambda x: x + 2**15)

    # if header['num_aux_input_channels']:
    #     fn('auxiliary.dat', 'aux_input_data', lambda x: x[::4])

    # if header['num_supply_voltage_channels']:
    #     fn('supply.dat', 'supply_voltage_data', lambda x: x[::128])

    # if header['num_board_adc_channels']:
    #     fn('analogin.dat', 'board_adc_data')

    if header['num_board_dig_in_channels']:
        fn('digitalin.dat', 'board_dig_in_raw')

    if header['num_board_dig_out_channels']:
        fn('digitalout.dat', 'board_dig_out_raw')

    return data

def parse_raw_data(header, data, rescale_to_volt):
    logging.info('Parsing data...')

    # Extract digital input channels to separate variables.
    for i in range(header['num_board_dig_in_channels']):
        data['board_dig_in_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_in_raw'], (1 << header['board_dig_in_channels'][i]['native_order'])), 0)

    # Extract digital output channels to separate variables.
    for i in range(header['num_board_dig_out_channels']):
        data['board_dig_out_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_out_raw'], (1 << header['board_dig_out_channels'][i]['native_order'])), 0)



    # NOTE(mwm) hack to decimate at some rate
    # DECIMATION_FACTOR = 20
    # # NOTE(mwm) do we want gaps?
    # data['t_amplifier'] = np.array(range(len(data['t_amplifier'][::DECIMATION_FACTOR])))
    # data['amplifier_data'] = data['amplifier_data'][:, ::DECIMATION_FACTOR]
    # data['aux_input_data'] = data['aux_input_data'][:, ::DECIMATION_FACTOR]
    # data['supply_voltage_data'] = data['supply_voltage_data'][:, ::DECIMATION_FACTOR]
    # data['temp_sensor_data'] = data['temp_sensor_data'][:, ::DECIMATION_FACTOR]
    # data['board_adc_data'] = data['board_adc_data'][:, ::DECIMATION_FACTOR]
    # data['board_dig_in_data'] = data['board_dig_in_data'][:, ::DECIMATION_FACTOR]
    # data['board_dig_in_raw'] = data['board_dig_in_raw'][::DECIMATION_FACTOR]
    # data['board_dig_out_data'] = data['board_dig_out_data'][:, ::DECIMATION_FACTOR]
    # data['board_dig_out_raw'] = data['board_dig_out_raw'][::DECIMATION_FACTOR]


    board_mode = header['eval_board_mode']
    print('This is the board mode: ' + str(board_mode))

    if rescale_to_volt:
        print('Scaling the raw values of the data to voltage values.')
        #Scale voltage levels appropriately.
        # data['amplifier_data'] = np.multiply(AMPLIFIER_TO_MICROVOLTS, (data['amplifier_data'].astype(np.int32) - 2**15))      # units = microvolts
        data['amplifier_data'] = np.multiply(AMPLIFIER_TO_MICROVOLTS, (data['amplifier_data'].astype(np.int32)))      # units = microvolts

        data['aux_input_data'] = np.multiply(AUXILIARY_TO_VOLTS, data['aux_input_data'])               # units = volts
        data['supply_voltage_data'] = np.multiply(SUPPLY_TO_VOLTS, data['supply_voltage_data'])     # units = volts

        if not board_mode:
            print('We are NOT in the board mode. ')
            data['board_adc_data'] = np.multiply(BOARD_MODE_ADC_TO_VOLTS_DEFAULT, data['board_adc_data'])           # units = volts
        else:
            print('We are IN the board mode. ')
            multiple = BOARD_MODE_ADC_TO_VOLTS[board_mode]
            data['board_adc_data'] = np.multiply(multiple, (data['board_adc_data'].astype(np.int32) - 32768)) # units = volts

    else:
        print('Using the raw values of the data, i.e. not scaling to voltage values. Not doing (x - 2**15) * 0.195. Just x.')
        
        data['amplifier_data'] = data['amplifier_data'].astype(np.int32)
        data['board_adc_data'] = data['board_adc_data'].astype(np.int32) - 32768 # units = volts


        check_time = 0
        if check_time: 
            print('Check the execution time.')

            start_time = time.time()        
            tmp = (data['board_adc_data'].astype(np.int32) - 2**15)      # units = microvolts
            end_time = time.time()        
            execution_time = end_time - start_time
            print(f"data['board_adc_data'].astype(np.int32) - 2**15: Execution time: {execution_time} seconds")

            
            start_time = time.time()        
            tmp = (data['board_adc_data'].astype(np.int32) - 32768)      # units = microvolts
            end_time = time.time()        
            execution_time = end_time - start_time
            print(f"(data['board_adc_data'].astype(np.int32) - 32768)  Execution time: {execution_time} seconds")

            

            start_time = time.time()        
            tmp = (data['board_adc_data'].astype(np.int32))      # units = microvolts
            end_time = time.time()        
            execution_time = end_time - start_time
            print(f"(data['board_adc_data'].astype(np.int32)) Execution time: {execution_time} seconds")

            start_time = time.time()        
            tmp = np.int32(data['board_adc_data'] - 32768)      # units = microvolts
            end_time = time.time()        
            execution_time = end_time - start_time
            print(f"np.int32(data['board_adc_data'] - 32768) Execution time: {execution_time} seconds")


            start_time = time.time()        
            tmp = np.int16(data['board_adc_data'] - 32768)      # units = microvolts
            # np.int16(data_intan_audio)
            end_time = time.time()        
            execution_time = end_time - start_time
            print(f"np.int16(data['board_adc_data'] - 32768) Execution time: {execution_time} seconds")


            start_time = time.time()        
            tmp = data['board_adc_data'].astype(np.int32) # units = volts
            # np.int16(data_intan_audio)
            end_time = time.time()        
            execution_time = end_time - start_time
            print(f"tmp = data['board_adc_data'].astype(np.int32)  Execution time: {execution_time} seconds")

            
            start_time = time.time()        
            data['board_adc_data'] = data['board_adc_data'].astype(np.int32) # units = volts
            # np.int16(data_intan_audio)
            end_time = time.time()        
            execution_time = end_time - start_time
            print(f"data['board_adc_data'] = data['board_adc_data'].astype(np.int32)  Execution time: {execution_time} seconds")

            # data['board_adc_data'] = data['board_adc_data'].astype(np.int32) - 2**15 # units = volts


    data['temp_sensor_data'] = np.multiply(TEMP_TO_C, data['temp_sensor_data'])               # units = deg C
    
    print('Convert to floating point values.')
    logging.debug("Convert to floating point values.")
    # data['amplifier_data'] = data['amplifier_data'].astype(np.float64)
    # data['aux_input_data'] = data['aux_input_data'].astype(np.float64)
    # data['supply_voltage_data'] = data['supply_voltage_data'].astype(np.float64)
    # data['board_adc_data'] = data['board_adc_data'].astype(np.float64)
    # data['temp_sensor_data'] = data['temp_sensor_data'].astype(np.float64)
    start_time = time.time()         
    data['amplifier_data'] = data['amplifier_data'].astype(np.float32)
    data['aux_input_data'] = data['aux_input_data'].astype(np.float32)
    data['supply_voltage_data'] = data['supply_voltage_data'].astype(np.float32)
    data['board_adc_data'] = data['board_adc_data'].astype(np.float32)
    data['temp_sensor_data'] = data['temp_sensor_data'].astype(np.float32)
    end_time = time.time()        
    execution_time = end_time - start_time
    print(f"Conversion to float: Execution time: {execution_time} seconds")

    print("Complete converting to floating point values.")


    print('Check for gaps in timestamps.')
    # Check for gaps in timestamps.
    print("Checking for gaps in timestamps.")
    
    start_time = time.time()        
    num_gaps = np.sum(np.not_equal(data['t_amplifier'][1:]-data['t_amplifier'][:-1], 1))
    if num_gaps == 0:
        logging.info('No missing timestamps in data.')
    else:
        logging.info('Warning: {0} gaps in timestamp data found.  Time scale will not be uniform!'.format(num_gaps))
        print('Warning: {0} gaps in timestamp data found.  Time scale will not be uniform!'.format(num_gaps))
        
    end_time = time.time()        
    execution_time = end_time - start_time
    print(f"Check for gap: Execution time: {execution_time} seconds")
    print("Complete checking for gaps in timestamps.")

        
    print('Scale time steps.')
    # Scale time steps (units = seconds).
    start_time = time.time()        
    data['t_amplifier'] = data['t_amplifier'] / header['sample_rate']
    data['t_aux_input'] = data['t_amplifier'][range(0, len(data['t_amplifier']), 4)]
    data['t_supply_voltage'] = data['t_amplifier'][range(0, len(data['t_amplifier']), header['num_samples_per_data_block'])]
    data['t_board_adc'] = data['t_amplifier']
    data['t_dig'] = data['t_amplifier']
    data['t_temp_sensor'] = data['t_supply_voltage']
 
    end_time = time.time()        
    execution_time = end_time - start_time
    print(f"scale time stamps: Execution time: {execution_time} seconds")

    # If the software notch filter was selected during the recording, apply the
    # same notch filter to amplifier data here.
    if header['notch_filter_frequency'] > 0 and header['version']['major'] < 3:
        logging.info('Applying notch filter...')
        print('Starting the notch filter application..')
        start_time = time.time()        

        increment = 10
        percent_done = increment
        for i in range(header['num_amplifier_channels']):
            data['amplifier_data'][i,:] = notch_filter(data['amplifier_data'][i,:], header['sample_rate'], header['notch_filter_frequency'], 10)

            fraction_done = 100 * (i / header['num_amplifier_channels'])
            if fraction_done >= percent_done:
                logging.info('{}% done...'.format(percent_done))
                percent_done += increment

        end_time = time.time()        
        execution_time = end_time - start_time
        print(f"Notch filter: Execution time: {execution_time} seconds")
        print('Complete applying the notch filter.')

    print('Complete parse_raw_data().')



def parse_raw_data_digital_only(header, data):
    logging.info('Parsing data...')

    # Extract digital input channels to separate variables.
    for i in range(header['num_board_dig_in_channels']):
        data['board_dig_in_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_in_raw'], (1 << header['board_dig_in_channels'][i]['native_order'])), 0)

    # Extract digital output channels to separate variables.
    for i in range(header['num_board_dig_out_channels']):
        data['board_dig_out_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_out_raw'], (1 << header['board_dig_out_channels'][i]['native_order'])), 0)

    # Scale voltage levels appropriately.
    # data['amplifier_data'] = np.multiply(AMPLIFIER_TO_MICROVOLTS, (data['amplifier_data'].astype(np.int32) - 2**15))      # units = microvolts
    # data['aux_input_data'] = np.multiply(AUXILIARY_TO_VOLTS, data['aux_input_data'])               # units = volts
    # data['supply_voltage_data'] = np.multiply(SUPPLY_TO_VOLTS, data['supply_voltage_data'])     # units = volts

    board_mode = header['eval_board_mode']
    # if not board_mode:
    #     data['board_adc_data'] = np.multiply(BOARD_MODE_ADC_TO_VOLTS_DEFAULT, data['board_adc_data'])           # units = volts
    # else:
    #     multiple = BOARD_MODE_ADC_TO_VOLTS[board_mode]
    #     data['board_adc_data'] = np.multiply(multiple, (data['board_adc_data'].astype(np.int32) - 2**15)) # units = volts

    # data['temp_sensor_data'] = np.multiply(TEMP_TO_C, data['temp_sensor_data'])               # units = deg C

    # Check for gaps in timestamps.
    # num_gaps = np.sum(np.not_equal(data['t_amplifier'][1:]-data['t_amplifier'][:-1], 1))
    # if num_gaps == 0:
    #     logging.info('No missing timestamps in data.')
    # else:
    #     logging.info('Warning: {0} gaps in timestamp data found.  Time scale will not be uniform!'.format(num_gaps))

    # # Scale time steps (units = seconds).
    # data['t_amplifier'] = data['t_amplifier'] / header['sample_rate']
    # data['t_aux_input'] = data['t_amplifier'][range(0, len(data['t_amplifier']), 4)]
    # data['t_supply_voltage'] = data['t_amplifier'][range(0, len(data['t_amplifier']), header['num_samples_per_data_block'])]
    # data['t_board_adc'] = data['t_amplifier']
    # data['t_dig'] = data['t_amplifier']
    # data['t_temp_sensor'] = data['t_supply_voltage']

    # # If the software notch filter was selected during the recording, apply the
    # # same notch filter to amplifier data here.
    # if header['notch_filter_frequency'] > 0 and header['version']['major'] < 3:
    #     logging.info('Applying notch filter...')

    #     increment = 10
    #     percent_done = increment
    #     for i in range(header['num_amplifier_channels']):
    #         data['amplifier_data'][i,:] = notch_filter(data['amplifier_data'][i,:], header['sample_rate'], header['notch_filter_frequency'], 10)

    #         fraction_done = 100 * (i / header['num_amplifier_channels'])
    #         if fraction_done >= percent_done:
    #             logging.info('{}% done...'.format(percent_done))
    #             percent_done += increment



def read_data(filename, dat=False, rescale_to_voltage = False):
    """Reads Intan Technologies RHD2000 data file generated by evaluation board GUI.

    Data are returned in a dictionary, for future extensibility.
    """
    tic = time.time()

    with open(filename, 'rb') as fid:

        header = read_header(fid)

        logging.info('Found {} amplifier channel{}.'.format(header['num_amplifier_channels'], plural(header['num_amplifier_channels'])))
        logging.info('Found {} auxiliary input channel{}.'.format(header['num_aux_input_channels'], plural(header['num_aux_input_channels'])))
        logging.info('Found {} supply voltage channel{}.'.format(header['num_supply_voltage_channels'], plural(header['num_supply_voltage_channels'])))
        logging.info('Found {} board ADC channel{}.'.format(header['num_board_adc_channels'], plural(header['num_board_adc_channels'])))
        logging.info('Found {} board digital input channel{}.'.format(header['num_board_dig_in_channels'], plural(header['num_board_dig_in_channels'])))
        logging.info('Found {} board digital output channel{}.'.format(header['num_board_dig_out_channels'], plural(header['num_board_dig_out_channels'])))
        logging.info('Found {} temperature sensors channel{}.'.format(header['num_temp_sensor_channels'], plural(header['num_temp_sensor_channels'])))
        logging.info('')

        filesize = os.path.getsize(filename)
        bytes_remaining = filesize - fid.tell()

        data = {}

        if dat:
            print('Reading the dat file...')
            data = read_dat_data(header, filename, rescale_to_voltage)
            # data = read_dat_data_digital_only(header, filename)
        elif bytes_remaining:
            data = read_rhd_data(header, fid, filesize)

    if data:
        parse_raw_data(header, data, rescale_to_voltage)
        # parse_raw_data_digital_only(header, data)

    # Move variables to result struct.
    result = data_to_result(header, data, bool(data))

    logging.info('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))

    return result

def plural(n):
    """Utility function to optionally pluralize words based on the value of n.
    """

    if n == 1:
        return ''
    else:
        return 's'

if __name__ == '__main__':
    a=read_data(sys.argv[1])
    #print(a)
