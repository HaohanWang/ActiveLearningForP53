package learnerHelper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
	int attributesNum = 100;

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

	public void selectInformativeAttributes() {
		trainingSet = null;
		testingSet = null;
		attributesNum = attributesNum-1;
		AttributeSelection attributeSelection = new AttributeSelection();
		InfoGainAttributeEval infoGainAttributeEval = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(attributesNum);
		attributeSelection.setEvaluator(infoGainAttributeEval);
		attributeSelection.setSearch(ranker);
		// attributeSelection.setInputFormat(trainingAllSet);
		int[] result = null;
		try {
			attributeSelection.SelectAttributes(trainingAllSet);
			result = attributeSelection.selectedAttributes();
		} catch (Exception e) {
			System.err.println("FAILED TO SELECT ATTRIBUTES");
			e.printStackTrace();
		}
		Arrays.sort(result);
		int i = 0;
		ArrayList<Integer> indexes = new ArrayList<Integer>();
		int j = 0;
		while (j < trainingAllSet.numAttributes()-1) {
			if (i < result.length && j == result[i]) {
				i += 1;
			} else {
				indexes.add(j);
			}
			j += 1;
		}
		trainingSet = new Instances(trainingAllSet);
		testingSet = new Instances(testingAllSet);
		for (int k = indexes.size() - 1; k >= 0; k--) {
			int index = indexes.get(k);
			if (index != trainingAllSet.classIndex()) {
				trainingSet.deleteAttributeAt(index);
				testingSet.deleteAttributeAt(index);
			}
		}
	}
	
	public void increaseTrainingSet(int index){
		trainingAllSet.add(testingAllSet.instance(index));
		testingAllSet.delete(index);
		selectInformativeAttributes();
	}
	
	public void increaseTrainingSetBatch(ArrayList<Integer> sortedResult){
		for (int i = sortedResult.size()-1;i>=0;i--){
			trainingAllSet.add(testingAllSet.instance(i));
			testingAllSet.delete(i);
		}
		selectInformativeAttributes();
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

	public int getAttributesNum() {
		return attributesNum;
	}

	public void setAttributesNum(int attributesNum) {
		this.attributesNum = attributesNum;
	}
	
	public static void main(String args[]) {
		DataManager dm = DataManager.getInstance();
		dm.selectInformativeAttributes();
		dm.setAttributesNum(100);
		Instances a = dm.getTrainingSet();
		Instances b = dm.getTestingSet();
		for (int k=0;k<=a.numInstances()-1; k++){
			if (a.instance(k).classValue() == 0.0){
				System.out.println(k);
			}
		}
	}
}
