from schedule import sched_update
from schedule_features import sched_features_update

def ui_update_schedule_actuals_data():
    print('Updating actual schedule data...')

    sched_update()

    print('Actual schedule data updated.\n')