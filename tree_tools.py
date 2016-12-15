#!/usr/bin/env python

'''
Utilities for parsing, outputing and understanding sklearn Trees.
'''
import numpy as np
import os
import sys

from collections import defaultdict
from copy import deepcopy

from sklearn.externals import joblib
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

__all__ = ["DecisionTreeClassifier",
           "DecisionTreeRegressor",
           "ExtraTreeClassifier",
           "ExtraTreeRegressor"]

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

LTE_OP = '<='
GT_OP  = '> '
ONE_HOT_MARK = '::'

class Pattern:
    '''
    Used to simplify rules encoded in trees. For instance, the rule "X > 10 and X > 2"
    will be simplified to "X > 2".
    '''
    def __init__(self):
      self.features = []
      self.one_hot_includes = defaultdict(set) # map feature name to set of values
      self.one_hot_excludes = defaultdict(set) # map feature name to set of values
      self.mins = {} # map feature name to a min (possible non existent)
      self.maxs = {} # map feature name to a max (possible non existent)

    def __repr__(self):
      rep = ''
      for f in self.features:
         vs = self.one_hot_includes.get(f)
         if vs is not None:
            rep += ' and %s in %s' % (f, list(vs))

         ''' only had excludes if the include list is empty '''
         if not self.one_hot_includes:
            vs = self.one_hot_excludes.get(f)
            if vs is not None:
               rep += ' and %s not in %s' % (f, list(vs))

         mn = self.mins.get(f)
         if mn is not None:
            if int(mn) == mn:
               mn_rep = mn
            else:
               mn_rep = '%+0.2f' % (mn,)
            rep += ' and %s%s%s' % (f, GT_OP, mn_rep)

         mx = self.maxs.get(f)
         if mx is not None:
            if int(mx) == mx:
               mx_rep = mx
            else:
               mx_rep = '%+0.2f' % (mx,)
            rep += ' and %s%s%s' % (f, LTE_OP, mx_rep)

      if rep[:4] == ' and':
         rep = rep[4:]
      rep = rep.replace('[','(')
      rep = rep.replace(']',')')
      return rep


    def add_condition(self, feature, op, t):
      if ONE_HOT_MARK in feature:
         ''' special handling for one-hot encoded variables
             if comp is '<=' then put in <exclude> set, 
             otherwise put in <include> set
         '''
         k,v = feature.split(ONE_HOT_MARK)
         if k not in self.features: # make sure order of insertion is maintained
            self.features.append(k)
         if op == LTE_OP:
            self.one_hot_excludes[k].add(v)
         else:
            self.one_hot_includes[k].add(v)
      else:
         if feature not in self.features: # make sure order of insertion is maintained
            self.features.append(feature)
         if op == LTE_OP:
            mx = self.maxs.get(feature)
            if mx is None:
               self.maxs[feature] = t
            else:
               self.maxs[feature] = min(t, mx)
         else:
            mn = self.mins.get(feature)
            if mn is None:
               self.mins[feature] = t
            else:
               self.mins[feature] = max(t, mn)
      return self


def tree_str(clf, feature_names=None, outfile=sys.stdout, full=True,
             max_depth=None, format_str="%+12.6f"):
   '''
   Parameters
   ----------
   clf : decision tree classifier
   The decision tree to be exported to graphviz.
   
   feature_names : list of strings, optional (default=None)
        Names of each of the features.
   
   Returns
   -------
   out_str: string
   The string to which the tree was printed.
   '''

   def node_to_str(tree, node_id):
      value = tree.value[node_id]
      impurity = tree.impurity[node_id]
      nns = tree.n_node_samples[node_id]
      if tree.n_outputs == 1:
         value = value[0, :]

      if hasattr(value, '__iter__') and len(value) > 1:
         # classifier impurity is returned, as well as value == [negatives, positives]
         if value[0] == 0 and value[1] == 0:
            rate = -1
         else:
            rate = value[1] / value.sum()

         val_str = ''.join("%8d" % _ for _ in value)
         # impurity monotonically increases with rate; not useful to see.
         # perf = "%8.6f %20s %6s  " % (rate, val_str, nns)
         perf = (format_str+" %6s  ") % (rate, nns)
      else:
         perf = (format_str+" %6s  ") % (value, nns)
      return perf

   def format_rule(feature, comp, t):
      if int(t) == t: # use int formatting
         return ["%s%2s%d" % (feature, comp, int(t))]
      return ["%s%2s%0.2f" % (feature, comp, t)]

   def recurse(tree, node_id, parent_id=None, pattern=Pattern(), depth=0, max_depth=None):
      if node_id == _tree.TREE_LEAF:
         raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

      left_child_id  = tree.children_left[node_id]
      right_child_id = tree.children_right[node_id]

      perf = node_to_str(tree, node_id)

      if left_child_id == _tree.TREE_LEAF or full == True:  # and right_child_id != _tree.TREE_LEAF
         #print >> outfile, perf + '  '.join(prefix)
         print >> outfile, perf + str(pattern)

      if depth >= max_depth:
        return

      if left_child_id != _tree.TREE_LEAF:  # and right_child_id != _tree.TREE_LEAF
         if feature_names is not None:
            feature = feature_names[tree.feature[node_id]]
            used_features.add(feature)
         else:
            feature = "X[%s]" % tree.feature[node_id]
         t = tree.threshold[node_id]

         #rule = format_rule(feature, LTE_OP, t)
         recurse(tree, left_child_id, node_id, deepcopy(pattern).add_condition(feature, LTE_OP, t), depth=depth+1, max_depth=max_depth)  # DFS (follow left-hand branches until leaf is encountered)

         #rule = format_rule(feature, GT_OP,  t)
         recurse(tree, right_child_id, node_id, deepcopy(pattern).add_condition(feature, GT_OP, t), depth=depth+1, max_depth=max_depth)

   used_features = set()

   if isinstance(clf, _tree.Tree):
      print >> outfile, '<tree>'
      recurse(clf, 0, Pattern(), depth=0, max_depth=max_depth)
      print >> outfile, '</tree>'
   elif hasattr(clf, 'estimators_'):
      for _ in clf.estimators_:
         tree_str(_, feature_names, outfile, full, max_depth, format_str)
   elif hasattr(clf, "tree_"):
      recurse(clf.tree_, 0, Pattern(), depth=0, max_depth=max_depth)
   else:
    print "No 'tree_' member found for ", clf


def get_feature_importances(clf, feature_names):
    assert len(clf.feature_importances_) == len(feature_names)
    total_importance = np.sum(clf.feature_importances_)
    imp_weights = clf.feature_importances_ / total_importance
    return sorted(zip(imp_weights, feature_names), reverse=True)


def dump_tree(clf, feature_names, fn='model_tree.txt', dir='', fop='w', 
    binaries=False, max_depth=4, format_str="%12.6f"):
    '''
    Persists a trained Tree (e.g. RandomForest) to text and numpy binary files.
    '''
    assert len(clf.feature_importances_) == len(feature_names)

    fn = str(fn)
    if binaries:
        joblib.dump(clf, os.path.join(dir, 'model_binary.' + fn))
        joblib.dump(feature_names, os.path.join(dir, 'model_feature_names.' + fn))

    full_path = os.path.join(dir, 'model_features_imps.' + fn + '.txt')
    with open(full_path, fop) as f:
        imps = get_feature_importances(clf, feature_names)
        print >> f, '\n'.join("%0.5f %s" % (imp,name) for imp, name in imps if imp >= 0)

    full_path = os.path.join(dir, 'model_trees.' + fn)
    with open(full_path, fop) as f:
        tree_str(clf, feature_names=feature_names, outfile=f, full=True, max_depth=max_depth, format_str=format_str)

    
def test_tree():
    '''
    Generate a tree for testing.
    '''
    # xn features
    xn = 5
    features = ['F'+ONE_HOT_MARK+'%d' % (_,) for _ in range(xn)]

    # first [0,xn/2] features contribute as i^2, last xn/2 don't contribute
    values = [_ if _ < xn/2 else 0 for _ in range(1,xn+1)]
    #values   = [pow(2,_) if _ < xn/2 else 0 for _ in range(xn)]

    print zip(features,values)

    n = 1000
    xs = np.zeros((n, xn))
    ys = np.zeros((n,))
    for i in range(n):
      bits = np.random.choice(values, size=xn/2+1, replace=False)
      for b in bits:
         xs[i][b-1] = 1
         ys[i] += values[b-1]
         #ys[i] += values[b-1]

    ys = np.array([np.sign(_) for _ in ys])

    for i in range(n):
      print "_ %5d%s" % (ys[i], xs[i])
    clf = RandomForestClassifier(n_estimators=1, min_samples_leaf=2, min_samples_split=2, bootstrap=True )
    #clf = RandomForestRegressor(n_estimators=1, min_samples_leaf=1, min_samples_split=1, )
    return clf.fit(xs, ys), features


def main():
   import argparse
   parser = argparse.ArgumentParser(description='read tree from sklearn model file, write to txt file')
   parser.add_argument('--test_tree', action='store_true')
   parser.add_argument('--path_to_model')
   parser.add_argument('--path_to_feature_names')
   args = parser.parse_args()

   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      exit(0)
   print args

   if args.test_tree:
      model, names = test_tree()
   else:
      model = joblib.load(args.path_to_model)
      names = joblib.load(args.path_to_feature_names)

   dump_tree(model, names, 'test_tree')

if __name__ == '__main__':
    main()
