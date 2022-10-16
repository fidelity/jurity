"""

    Template script for the submission. You can use this as a starting point for your code: you can
    copy this script as is into your repository, and then modify the associated Model class to include
    your logic, instead of the random baseline. Most of this script should be left unchanged for your submission
    as we should be able to run your code to confirm your scores.

    Please make sure you read and understand the competition rules and guidelines before you start.

"""

import argparse
import os
from datetime import datetime
from dotenv import load_dotenv

# import env variables from file
load_dotenv('upload.env', verbose=True)

# variables for the submission
EMAIL = os.getenv('EMAIL')  # the e-mail you used to sign up
assert EMAIL != '' and EMAIL is not None
BUCKET_NAME = os.getenv('BUCKET_NAME')  # you received it in your e-mail
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID')  # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')  # you received it in your e-mail
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')  # you received it in your e-mail


def parse_args():
    parser = argparse.ArgumentParser(description='Run submission.')
    parser.add_argument('--factors', dest='factors', type=int, default=50)
    parser.add_argument('--regularization', dest='regularization', type=float, default=0.1)
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.1)
    parser.add_argument('--dont_upload', action='store_true')
    # one of `None`, `user_activity_fairness`, `track_activity_fairness`
    parser.add_argument('--post_proc_type', default='user_activity_fairness')
    # post_proc_num_popular = 200,  # only valid for `user_activity_fairness`
    # # The quantile value that a score has to beat to be called "positive"
    # # Higher is more selective
    # post_proc_positive_quantile = 0.8,
    # is_member_max_positive_activity_bin = 2,
    parser.add_argument('--use_average', action='store_true')

    cmd = parser.parse_args()
    args = vars(cmd)
    print(args)
    return args

# run the evaluation loop when the script is called directly
if __name__ == '__main__':
    # import the basic classes
    from evaluation.EvalRSRunner import EvalRSRunner
    from evaluation.EvalRSRecList import FairnessRecList
    from evaluation.EvalRSRunner import ChallengeDataset
    from submission.MyModel import MyModel
    print('\n\n==== Starting evaluation script at: {} ====\n'.format(datetime.utcnow()))
    # load the dataset
    print('\n\n==== Loading dataset at: {} ====\n'.format(datetime.utcnow()))
    # this will load the dataset with the default values for the challenge
    dataset = ChallengeDataset()
    print('\n\n==== Init runner at: {} ====\n'.format(datetime.utcnow()))
    # run the evaluation loop
    runner = EvalRSRunner(
        dataset=dataset,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        participant_id=PARTICIPANT_ID,
        bucket_name=BUCKET_NAME,
        email=EMAIL
        )
    print('==== Runner loaded, starting loop at: {} ====\n'.format(datetime.utcnow()))
    # NOTE: this evaluation will run with default values for the parameters and the upload flag
    # For local testing and iteration, you can check the tutorial in the notebooks folder and the
    # kaggle notebook: https://www.kaggle.com/code/vinidd/cikm-data-challenge

    args = parse_args()
    factors = args['factors']
    regularization = args['regularization']
    alpha = args['alpha']
    post_proc_type = args['post_proc_type']
    use_average = args['use_average']

    my_model = MyModel(
        items=dataset.df_tracks,
        users=dataset.df_users,
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        # kwargs may contain additional arguments in case, for example, you
        # have data augmentation functions that you wish to use in combination
        # with the dataset provided by the runner.
        # Post-process params
        post_proc_type=post_proc_type,  # one of `None`, `user_activity_fairness`, `track_activity_fairness`
        post_proc_num_popular=200,  # only valid for `user_activity_fairness`
        # The quantile value that a score has to beat to be called "positive"
        # Higher is more selective
        post_proc_positive_quantile=0.8,
        is_member_max_positive_activity_bin=2,
        use_average=use_average,
        my_custom_argument='my_custom_argument' 
    )
    # run evaluation with your model
    # the evaluation loop will magically perform the fold splitting, training / testing
    # and then submit the results to the leaderboard
    runner.evaluate(
        model=my_model,
        custom_RecList=FairnessRecList,
        upload=not args['dont_upload']
        )
    print('\n\n==== Evaluation ended at: {} ===='.format(datetime.utcnow()))
