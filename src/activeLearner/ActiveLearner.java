package activeLearner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Random;

import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import learnerHelper.DataManager;

public class ActiveLearner {
	DataManager dm = null;
	SMO classifier = null;
	double alpha = 1;
	int positiveFound = 0;
	int numberSelected = 5;
	int initPositive = 50;
	int initNegative = 620;
	int step = 0;

	public ActiveLearner() {
		dm = DataManager.getInstance();
		classifier = new SMO();
		dm.setAttributesNum(100);
		dm.selectInformativeAttributes();
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	// get maximum curiosity
	public double getMaximumCuriosity(Instances ts, Instance ins) {
		ins.setClassValue("active");
		ts.add(ins);
		double rp = getCuriosity(ts);
		ts.delete(ts.numInstances() - 1);
		ins.setClassValue("inactive");
		ts.add(ins);
		double rn = getCuriosity(ts);
		ts.delete(ts.numInstances() - 1);
		return Math.max(rp, rn);
	}

	// get curiosity
	public double getCuriosity(Instances ts) {
		ts.randomize(new Random());
		int mid = (int) (ts.numInstances() * 0.9);
		Instances train = new Instances(ts, 0, mid);
		Instances test = new Instances(ts, mid, ts.numInstances() - mid);
		double[] result = evaluateBasic(train, test);
		double tp = result[0];
		double fp = result[1];
		double tn = result[2];
		double fn = result[3];
		double r = ((tp * tn) - (fp * fn))
				/ Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
		return r;
	}

	// select next instances single mode
	public int selectNextInstance() {
		Instances testingSet = dm.getTestingSet();
		Instances trainingSet = dm.getTrainingSet();
		SMO localClassifier = new SMO();
		try {
			localClassifier.buildClassifier(trainingSet);
		} catch (Exception e) {
			System.err.println("FAILED TO BUILD LOCAL CLASSIFIER");
			e.printStackTrace();
		}
		double max = 0.0;
		int index = -1;
		int temp = 0;
		for (int i = 0; i < testingSet.numInstances(); i++) {
			double score = getMaximumCuriosity(trainingSet,
					testingSet.instance(i));
			try {
				if (localClassifier.classifyInstance(testingSet.instance(i)) == 0.0) {
					score += alpha;
					temp += 1;
				}
			} catch (Exception e) {
				System.err.println("FAILED TO CLASSIFY AN INSTANCE");
				e.printStackTrace();
			}
			if (score > max) {
				index = i;
				max = score;
			}
		}
		System.out.print(" curiosity: " + max);
		System.out.print("found " + temp);
		if (testingSet.instance(index).classValue() == 0.0)
			positiveFound += 1;
		System.out.print(" positive Found: " + positiveFound);
		return index;
	}

	public ArrayList<Integer> selectNextInstanceBatch() {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Instances testingSet = dm.getTestingSet();
		Instances trainingSet = dm.getTrainingSet();
		LinkedList<Tuple> seq = new LinkedList<Tuple>();
		int capacity = numberSelected;
		SMO localClassifier = new SMO();
		try {
			localClassifier.buildClassifier(trainingSet);
		} catch (Exception e) {
			System.err.println("FAILED TO BUILD LOCAL CLASSIFIER");
			e.printStackTrace();
		}
		for (int i = 0; i < testingSet.numInstances(); i++) {
			Instances m = new Instances(testingSet, i, 1);
			double score = getMaximumCuriosity(trainingSet, m.instance(0));
			try {
				if (localClassifier.classifyInstance(testingSet.instance(i)) == 0.0) {
					double w = Math.pow((1-getPositiveSetPercentage()), 4);
					score += w;
//					System.out.println(w);
//					score += 1.0;
				}
			} catch (Exception e) {
				System.err.println("FAILED TO CLASSIFY AN INSTANCE");
				e.printStackTrace();
			}
			if (capacity != 0) {
				if (score > 0) {
					Tuple tmp = new Tuple(i, score);
					seq.add(tmp);
					capacity -= 1;
				}
				Collections.sort(seq, new Tuple());
			} else {
				if (score > seq.get(0).y) {
					Tuple tmp = new Tuple(i, score);
					seq.removeFirst();
					seq.add(tmp);
					Collections.sort(seq, new Tuple());
				}
			}
		}
		for (int i = 0; i < seq.size(); i++) {
			int index = seq.get(i).x;
			result.add(index);
			if (testingSet.instance(index).classValue() == 0.0)
				positiveFound += 1;
		}
		System.out.print(" positive Found " + positiveFound);
		Collections.sort(result);
		return result;
	}

	public double getPositiveSetPercentage() {
		return (double) (positiveFound + initPositive)
				/ (double) (numberSelected * step - positiveFound + initNegative);
	}

	// basic evaluate function
	public double[] evaluateBasic(Instances train, Instances test) {
		double[] result = new double[4];
		try {
			classifier.buildClassifier(train);
		} catch (Exception e) {
			System.err.println("FAILED TO BUILD CLASSIFIER");
			e.printStackTrace();
		}
		double tp = 0.0, tn = 0.0, fp = 0.0, fn = 0.0;
		try {
			for (int i = 0; i < test.numInstances(); i++) {
				if (classifier.classifyInstance(test.instance(i)) == 0.0
						&& test.instance(i).classValue() == 0.0) {
					tp += 1;
				}
				if (classifier.classifyInstance(test.instance(i)) == 0.0
						&& test.instance(i).classValue() == 1.0) {
					fp += 1;
				}
				if (classifier.classifyInstance(test.instance(i)) == 1.0
						&& test.instance(i).classValue() == 0.0) {
					fn += 1;
				}
				if (classifier.classifyInstance(test.instance(i)) == 1.0
						&& test.instance(i).classValue() == 1.0) {
					tn += 1;
				}
			}
		} catch (Exception e) {
			System.err.println("FAILED TO CLASSIFY AN INSTANCE");
			e.printStackTrace();
		}
		result[0] = tp;
		result[1] = fp;
		result[2] = tn;
		result[3] = fn;
		return result;
	}

	// Evaluate Function, get F-score
	public double evaluate() {
		Instances test = dm.getTestingSet();
		Instances train = dm.getTrainingSet();
		double[] result = evaluateBasic(train, test);
		double tp = result[0];
		double fp = result[1];
		double tn = result[2];
		double fn = result[3];
		double accuracy = (tp + tn) / (tp + tn + fp + fn);
		double precision = tp / (tp + fp);
		double recall = tp / (tp + fn);
		System.out.print(" Three scores & accuracy: " + tp + "\t" + fp + "\t"
				+ fn + "\t" + accuracy);
		System.out.print(" precision " + (double)(tp)/(tp+fp)+" recall "+(double)(tp)/(tp+fn));
		double F = (1 + alpha * alpha) * precision * recall
				/ (alpha * alpha * precision + recall);
		return F;
	}

	// Main Process Active Learning
	public ArrayList<Double> activeLearning() {
		boolean bacth = true;
		ArrayList<Double> result = new ArrayList<Double>();
		step = 1;
		int total = 100;
		while (step <= total) {
			System.out.print("Step " + step);
			if (bacth == false) {
				int index = selectNextInstance();
				dm.increaseTrainingSet(index);
			} else {
				ArrayList<Integer> batchIndex = selectNextInstanceBatch();
				dm.increaseTrainingSetBatch(batchIndex);
			}
			double F = evaluate();
			result.add(F);
			System.out.println(" Fscore " + F);
			step++;
		}
		return result;
	}

	static public void main(String args[]) {
		ActiveLearner al = new ActiveLearner();
		al.setAlpha(1);
		al.activeLearning();
	}

	public class Tuple implements Comparator<Tuple> {
		public final int x;
		public final double y;

		public Tuple(int x, double y) {
			this.x = x;
			this.y = y;
		}

		public Tuple() {
			this.x = 0;
			this.y = 0;
		}

		@Override
		public int compare(Tuple o1, Tuple o2) {
			return Double.compare(o1.y, o2.y);
		}
	}

	public void OutPutPriorityQueue(PriorityQueue<Tuple> heap) {
		Tuple[] temp = new Tuple[5];
		for (int i = 0; i < 5; i++) {
			temp[i] = heap.poll();
			System.out.print(temp[i].x + " " + temp[i].y + "\t");
		}
		System.out.println();
		for (int i = 0; i < 5; i++) {
			heap.add(temp[i]);
		}
	}
}
