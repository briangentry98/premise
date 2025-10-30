
import pandas as pd
import numpy as np
from pathlib import Path
# import yaml

def run_two_wheeler(scenario_name):
    # Need to change path for each scenario
    DATA_DIR = Path(r"../GCAM_queryresults_"+scenario_name)

    print(DATA_DIR)

    # load LCI data from GCAM for passenger car. two files: one with physical output (activity in passenger-km) and one with energy use (in EJ)
    two_wheeler_output = pd.read_csv(DATA_DIR /'two wheeler physical output by technology.csv')
    two_wheeler_input = pd.read_csv(DATA_DIR /'two wheeler final energy by technology and fuel.csv')


    # we need to reshape all of the data in a format premise can understand
    # first, reshape two_wheeler_output
    # store in temp_df
    temp_df = two_wheeler_output.copy()
    temp_df= temp_df.groupby(['Units', 'scenario', 'region', 'sector', 'subsector', 'technology','Year'])['value'].agg('sum').reset_index()
    # create out_df which will be written to file
    two_wheeler_output = temp_df.copy()
    # add world region by aggregating all data
    temp_df = temp_df.groupby(['Units', 'scenario', 'sector', 'subsector', 'technology','Year'])['value'].agg('sum').reset_index()
    temp_df['region'] = 'World'
    # concatenate dfs
    two_wheeler_output = pd.concat([two_wheeler_output, temp_df], axis=0)


    # now reshape two_wheeler_input
    temp_df = two_wheeler_input.copy()
    temp_df = temp_df.groupby(['Units', 'scenario', 'region', 'sector', 'subsector', 'technology','Year'])['value'].agg('sum').reset_index()
    two_wheeler_input = temp_df.copy()
    # add world region by aggregating all data
    temp_df = temp_df.groupby(['Units', 'scenario', 'sector', 'subsector', 'technology','Year'])['value'].agg('sum').reset_index()
    temp_df['region'] = 'World'
    two_wheeler_input = pd.concat([two_wheeler_input, temp_df], axis=0)

    # now we need to format these dfs into IAMC format
    # first, rename existing columns to columns in IAMC format
    two_wheeler_input = two_wheeler_input.rename(columns={'region': 'Region', 'scenario': 'Scenario', 'Units': 'Unit'})
    two_wheeler_output = two_wheeler_output.rename(columns={'region': 'Region', 'scenario': 'Scenario', 'Units': 'Unit'})

    # replace Scenario with scenario_name
    two_wheeler_input['Scenario'] = scenario_name
    two_wheeler_output['Scenario'] = scenario_name

    # add GCAM as Model
    two_wheeler_input['Model'] = 'GCAM'
    two_wheeler_output['Model'] = 'GCAM'

    # replace Unit column with expected values (EJ/yr, million tkm/yr)
    two_wheeler_input['Unit'] = 'EJ/yr'
    two_wheeler_output['Unit'] = 'million pkm/yr'

    # define Variable
    # input: Final Energy|Transport|Pass|Road|{subsector}|{technology}
    # output: Distance|Transport|Pass|Road|{subsector}|{technology}
    two_wheeler_input['Variable'] = 'Final Energy|Transport|Pass|Road|LDV|Two Wheelers|' + two_wheeler_input['technology']
    two_wheeler_output['Variable'] = 'Distance|Transport|Pass|Road|LDV|Two Wheelers|' + two_wheeler_output['technology'] 

    # reorder columns and remove unnecessary columns (sector, subsector, technology)
    two_wheeler_input = two_wheeler_input[['Scenario', 'Region', 'Model', 'Variable', 'Unit', 'Year', 'value']]
    two_wheeler_output = two_wheeler_output[['Scenario', 'Region', 'Model', 'Variable', 'Unit', 'Year', 'value']]

    # pivot dfs, creating columns for each year

    two_wheeler_input_pivot = pd.pivot_table(two_wheeler_input,
                                      values=['value'],
                                      index=['Scenario', 'Region', 'Model', 'Variable', 'Unit'],
                                      columns=['Year']).reset_index()
    two_wheeler_output_pivot = pd.pivot_table(two_wheeler_output,
                                       values=['value'],
                                       index=['Scenario', 'Region', 'Model', 'Variable', 'Unit'],
                                       columns=['Year']).reset_index()
    out_df = pd.concat([two_wheeler_input_pivot, two_wheeler_output_pivot]).reset_index(drop=True)

    # tidy up dataframe (fix multiple index column names in year columns)
    out_df.columns = ['Scenario', 'Region', 'Model', 'Variable', 'Unit'] + [str(x[1]) for x in out_df.columns[5:]]

    # write to file
    out_df.to_excel('./iamc_template/'+scenario_name+'/iamc_template_gcam_two_wheeler_world.xlsx', index=False)



run_two_wheeler('SSP2 RCP26')