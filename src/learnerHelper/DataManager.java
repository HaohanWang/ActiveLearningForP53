package learnerHelper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;

public class DataManager {
	private static DataManager instance = null;

	Instances trainingSet = null;
	Instances testingSet = null;
	Instances trainingAllSet = null;
	Instances testingAllSet = null;
	InfoGainAttributeEval eval = null;

	protected DataManager() {
		eval = new InfoGainAttributeEval();
		File inputFile = new File("data/data.arff");
		ArffLoader atf = new ArffLoader();
		Instances set = null;
		try {
			atf.setFile(inputFile);
			set = atf.getDataSet();
		} catch (IOException e) {
			System.err.println("FAILED TO GET INPUT FILES");
			e.printStackTrace();
		}
		trainingAllSet = new Instances(set, 0, 50);
		testingAllSet = new Instances(set, 50, 180);
		for (int i = 180; i <= 800; i++) {
			trainingAllSet.add(set.instance(i));
		}
		for (int i = 801; i <= 3000; i++) {
			testingAllSet.add(set.instance(i));
		}
		trainingAllSet.setClassIndex(trainingAllSet.numAttributes() - 1);
		testingAllSet.setClassIndex(trainingAllSet.numAttributes() - 1);
	}

	static public DataManager getInstance() {
		if (instance == null) {
			instance = new DataManager();
		}
		return instance;
	}

	public Instances getTrainingSet() {
		return trainingSet;
	}

	public Instances getTestingSet() {
		return testingSet;
	}

	public void selectInformativeAttributes(int num) {
//		ArrayList<Tuple> infolist = new ArrayList<Tuple>();
//		try {
//			eval.buildEvaluator(trainingAllSet);
//			for (int i = 0; i < trainingAllSet.numAttributes(); i++) {
//				double score = eval.evaluateAttribute(i);
//				//System.out.println(score);
//				Tuple temp = new Tuple(i, score);
//				infolist.add(temp);
//			}
//		} catch (Exception e) {
//			System.err.println("FAILED TO EVALUATING SCORE");
//			e.printStackTrace();
//		}
//		Collections.sort(infolist, new Tuple(0, 1));
//		trainingSet = new Instances(trainingAllSet);
//		testingSet = new Instances(testingAllSet);
//		ArrayList<Integer> indexes = new ArrayList<Integer>();
//		for (int i = num; i < infolist.size(); i++) {
//			indexes.add(infolist.get(i).x);
//		}
//		Collections.sort(indexes);
//		for (int i = indexes.size()-1; i >=0; i--) {
//			int index = indexes.get(i);
//			if (index != trainingAllSet.classIndex()) {
//				trainingSet.deleteAttributeAt(index);
//				testingSet.deleteAttributeAt(index);
//			}
//		}
		AttributeSelection attributeSelection = new AttributeSelection();
	    InfoGainAttributeEval infoGainAttributeEval = new InfoGainAttributeEval();
	    Ranker ranker = new Ranker();
	    ranker.setNumToSelect(num);
	    attributeSelection.setEvaluator(infoGainAttributeEval);
	    attributeSelection.setSearch(ranker);
	    attributeSelection.setInputFormat(trainingAllSet);
	    trainingSet = Filter.useFilter(trainingAllSet, attributeSelection);
	    testingSet = Filter.useFilter(testingAllSet, attributeSelection);
	}

	public class Tuple implements Comparator<Tuple> {
		public final int x;
		public final double y;

		public Tuple(int x, double y) {
			this.x = x;
			this.y = y;
		}

		@Override
		public int compare(Tuple o1, Tuple o2) {
			return (int) (o1.y - o2.y);
		}
	}

	public static void main(String args[]) {
		DataManager dm = DataManager.getInstance();
		dm.selectInformativeAttributes(100);
		Instances a = dm.getTrainingSet();
		Instances b = dm.getTestingSet();
		System.out.println(a.numAttributes());
		System.out.println(b.numInstances());
	}
}
