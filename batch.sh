set -x
set -e
# Any subsequent(*) commands which fail will cause the shell script to exit immediately

# time python eval.py --in_csv ~/scratch/grp1.ds.txt ~/scratch/prj2 --sym_fit LP;
# time python eval.py ~/scratch/prj2 --grp_fit LP --sym_fit LP;
# time python eval.py ~/scratch/prj2 --grp_fit RF;
time python eval.py ~/scratch/prj2 --grp_fit RF --input_time ; #--criterion mae;
time python eval.py ~/scratch/prj2 --grp_fit RF --input_extras ; #--criterion mae;
time python eval.py ~/scratch/prj2 --grp_fit RF --input_time --input_extras ; #--criterion mae;

time python eval.py ~/scratch/prj2 --sym_fit RF --input_time;
time python eval.py ~/scratch/prj2 --sym_fit RF --input_extras;
time python eval.py ~/scratch/prj2 --sym_fit RF --input_time --input_extras;

aws ec2 stop-instances --instance-ids i-0e77d63036144e3d8

