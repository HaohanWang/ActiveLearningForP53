package learnerHelper;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class DataManager {
	private static DataManager instance = null;

	Instances trainingSet = null;
	Instances testingSet = null;

	protected DataManager() {
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
		trainingSet = new Instances(set, 0, 50);
		testingSet = new Instances(set, 50, 180);
		for (int i = 180; i <= 800; i++) {
			trainingSet.add(set.instance(i));
		}
		for (int i = 801; i <= 3170; i++) {
			testingSet.add(set.instance(i));
		}
		trainingSet.setClassIndex(trainingSet.numAttributes() - 1);
		testingSet.setClassIndex(trainingSet.numAttributes() - 1);
	}

	public DataManager getInstance() {
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

	public static void main(String args[]) {
		DataManager dm = new DataManager().getInstance();
		Instances a = dm.getTrainingSet();
		Instances b = dm.getTestingSet();
		System.out.println(a.numInstances());
		System.out.println(b.numInstances());
	}
}
