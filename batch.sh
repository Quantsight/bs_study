set -x
# time python eval.py --in_csv ~/scratch/grp1.ds.txt ~/scratch/prj2 --sym_fit LP;
# time python eval.py ~/scratch/prj2 --grp_fit LP --sym_fit LP;
time python eval.py ~/scratch/prj2 --sym_fit RF;
# time python eval.py ~/scratch/prj2 --grp_fit LP;
time python eval.py ~/scratch/prj2 --grp_fit RF --no_RF_sym;
time python eval.py ~/scratch/prj2 --grp_fit RF;
time python eval.py ~/scratch/prj2 --grp_fit LP --sym_fit RF;
time python eval.py ~/scratch/prj2 --grp_fit RF --sym_fit LP;
time python eval.py ~/scratch/prj2 --grp_fit RF --sym_fit RF;

