package activeLearner;

import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import learnerHelper.DataManager;

public class ActiveLearner {
	DataManager dm = null;
	SMO classifier = null;
	double alpha = 0.5;
	
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

	public double getCuriosity(Instances ts) {
		ts.randomize(new Random());
		int mid = (int) (ts.numInstances() * 0.9);
		Instances train = new Instances(ts, 0, mid);
		Instances test = new Instances(ts, mid, ts.numInstances()-mid);
		double[] result = evaluateBasic(train, test);
		double tp = result[0];
		double fp = result[1];
		double tn = result[2];
		double fn = result[3];
		double r = ((tp * tn) - (fp * fn))
				/ Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
		return r;
	}

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
		for (int i = 0; i < testingSet.numInstances(); i++) {
			double score = getMaximumCuriosity(trainingSet,
					testingSet.instance(i));
			try {
				if (localClassifier.classifyInstance(testingSet.instance(i)) == 0.0){
					score += alpha;
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			if (score > max) {
				index = i;
				max = score;
			}
		}
		System.out.print(" curiosity: " + max);
		return index;
	}

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

	public double evaluate() {
		Instances test = dm.getTestingSet();
		Instances train = dm.getTrainingSet();
		double[] result = evaluateBasic(train, test);
		double tp = result[0];
		double fp = result[1];
		double fn = result[3];
		double precision = tp / (tp + fp);
		double recall = tp / (tp + fn);
		System.out.print(" Three scores: "+tp+"\t"+fp+"\t"+fn);
		double F = (1 + alpha * alpha) * precision * recall
				/ (alpha * alpha * precision + recall);
		return F;
	}

	public ArrayList<Double> activeLearning() {
		ArrayList<Double> result = new ArrayList<Double>();
		int step = 1;
		int total  = 100;
		while (step <= total) {
			System.out.print("Step "+step);
			int index = selectNextInstance();
			dm.increaseTrainingSet(index);
			double F = evaluate();
			result.add(F);
			System.out.println(" Fscore"+F);
			step++;
		}
		return result;
	}
	
	static public void main(String args[]){
		ActiveLearner al = new ActiveLearner();
		al.setAlpha(0.75);
		al.activeLearning();
	}
}
