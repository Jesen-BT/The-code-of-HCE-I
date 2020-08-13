package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.gui.experimentertab.Summary;
import moa.options.ClassOption;

import javax.swing.*;
import java.util.List;

public class HOB extends AbstractClassifier implements MultiClassClassifier {
    private static final long serialVersionUID = 1L;

    public FloatOption theta = new FloatOption("theta", 't', "The time decay factor for class size.", 0.9, 0, 1);
    public FloatOption alpha = new FloatOption("alpha", 'a', "Undersampling factor.", 0.9, 0, 1);
    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class, "drift.DriftDetectionMethodClassifier -l trees.HoeffdingTree");
    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's', "The number of models in the bag.", 5, 1, 2147483647);
    public IntOption passEnsembleSizeOption = new IntOption("passEnsembleSize", 'q', "the number of passive learner", 5, 1, 2147483647);
    public IntOption chunkSizeOption = new IntOption("chunkSize", 'c', "The chunk size used for classifier creation and evaluation.", 500, 1, Integer.MAX_VALUE);
    public ClassOption passiveLearner = new ClassOption("passiveLearner", 'p', "the passive learner", Classifier.class, "trees.HoeffdingTree");
    protected Classifier[] ensemble;
    protected Classifier[] passEnsemble;
    protected Instance[] buffer;
    protected int iterationControl = 0;
    protected int currentBaseLearnerNo = -1;
    protected double activeWeight[];
    protected double passiveWeight[];
    protected double classSize[];
    protected int numEnsembleLearners;


    public String getPurposeString() {
        return "Oversampling on-line bagging of Wang et al IJCAI 2016.";
    }

    public HOB() {
        classSize = null;
    }

    public void resetLearningImpl() {
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier)this.getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        this.activeWeight = new double[this.ensembleSizeOption.getValue()];
        for(int i = 0; i < this.ensemble.length; ++i) {
            this.ensemble[i] = baseLearner.copy();
            this.activeWeight[i] = 1/(double)(this.ensembleSizeOption.getValue()+this.passEnsembleSizeOption.getValue());
        }

        this.passEnsemble = new Classifier[this.passEnsembleSizeOption.getValue()];
        Classifier passive = (Classifier)this.getPreparedClassOption(this.passiveLearner);
        passive.resetLearning();
        this.passiveWeight = new double[this.passEnsembleSizeOption.getValue()];
        for (int i = 0; i < this.passEnsemble.length; i++) {
            this.passEnsemble[i] = passive.copy();
            this.passiveWeight[i] = 1/(double)(this.ensembleSizeOption.getValue()+this.passEnsembleSizeOption.getValue());
        }
        iterationControl = 0;
        currentBaseLearnerNo=-1;
        this.buffer = new Instance[chunkSizeOption.getValue()];
        this.numEnsembleLearners = this.ensembleSizeOption.getValue()+this.passEnsembleSizeOption.getValue();
    }


    public void trainOnInstanceImpl(Instance inst) {

        updateClassSize(inst);
        double lambda = calculatePoissonLambda(inst);

        for (int i = 0; i < this.ensemble.length; i++) {
            int k = MiscUtils.poisson(lambda, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) inst.copy();
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
            }
            if (ensemble[i].correctlyClassifies(inst)){
                activeWeight[i] *= ((double)1 + (1/numEnsembleLearners)*(1-classSize[(int)inst.classValue()]));
            }else {
                activeWeight[i] *= ((double)1 - (1/numEnsembleLearners)*(1-classSize[(int)inst.classValue()]));
            }
        }

        for (int i = 0; i < this.passEnsemble.length; i++) {
            int k = MiscUtils.poisson(lambda, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) inst.copy();
                weightedInst.setWeight(inst.weight() * k);
                this.passEnsemble[i].trainOnInstance(weightedInst);
            }

            if (passEnsemble[i].correctlyClassifies(inst)){
                passiveWeight[i] *= ((double)1 + (1/numEnsembleLearners)*(1-classSize[(int)inst.classValue()]));
            }else{
                passiveWeight[i] *= ((double)1 - (1/numEnsembleLearners)*(1-classSize[(int)inst.classValue()]));
            }
        }

        Normalized();

        buffer[iterationControl] = inst;
        iterationControl = (iterationControl+1)%this.chunkSizeOption.getValue();
        if (iterationControl==0){
            creatNewBaseLearner();
        }

    }

    public void creatNewBaseLearner(){

        currentBaseLearnerNo = (currentBaseLearnerNo+1)%this.passEnsembleSizeOption.getValue();
        passEnsemble[currentBaseLearnerNo].resetLearning();

        for(Instance inst: buffer){
            passEnsemble[currentBaseLearnerNo].trainOnInstance(inst);
        }

        passiveWeight[currentBaseLearnerNo] = 1/(double)(numEnsembleLearners);
        Normalized();
    }

    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();

        for(int i = 0; i < this.ensemble.length; ++i) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            if (vote.sumOfValues() > 0.0D) {
                vote.normalize();
                vote.scaleValues(this.activeWeight[i]);
                combinedVote.addValues(vote);
            }
        }

        for (int i = 0; i < this.passEnsemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.passEnsemble[i].getVotesForInstance(inst));
            if (vote.sumOfValues() > 0.0D) {
                vote.normalize();
                vote.scaleValues(this.passiveWeight[i]);
                combinedVote.addValues(vote);
            }
        }





        return combinedVote.getArrayRef();
    }

    protected void updateClassSize(Instance inst) {
        if (this.classSize == null) {
            classSize = new double[inst.numClasses()];

            // start class size as equal for all classes
            for (int i=0; i<classSize.length; ++i) {
                classSize[i] = 1d/classSize.length;
            }
        }

        for (int i=0; i<classSize.length; ++i) {
            classSize[i] = theta.getValue() * classSize[i] + (1d - theta.getValue()) * ((int) inst.classValue() == i ? 1d:0d);
        }
    }

    // classInstance is the class corresponding to the instance for which we want to calculate lambda
    public double calculatePoissonLambda(Instance inst) {
        double lambda = 1d;
        int majClass = getMajorityClass();

        lambda = (classSize[majClass]*alpha.getValue()) / classSize[(int) inst.classValue()];

        return lambda;
    }

    // will result in an error if classSize is not initialised yet
    public int getMajorityClass() {
        int indexMaj = 0;

        for (int i=1; i<classSize.length; ++i) {
            if (classSize[i] > classSize[indexMaj]) {
                indexMaj = i;
            }
        }
        return indexMaj;
    }

    // will result in an error if classSize is not initialised yet
    public int getMinorityClass() {
        int indexMin = 0;

        for (int i=1; i<classSize.length; ++i) {
            if (classSize[i] <= classSize[indexMin]) {
                indexMin = i;
            }
        }
        return indexMin;
    }

    public void Normalized(){
        double sum = 0;
        for (int i = 0; i < activeWeight.length; i++) {
            sum += activeWeight[i];
        }
        for (int i = 0; i < passiveWeight.length; i++) {
            sum += passiveWeight[i];
        }
        for (int i = 0; i < activeWeight.length; i++) {
            activeWeight[i] =  activeWeight[i]/sum;
        }
        for (int i = 0; i < passiveWeight.length; i++) {
            passiveWeight[i] = passiveWeight[i]/sum;
        }
    }


    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size", this.ensemble != null ? (double)this.ensemble.length+this.passEnsemble.length : 0.0D)};
    }

    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return true;
    }

}
