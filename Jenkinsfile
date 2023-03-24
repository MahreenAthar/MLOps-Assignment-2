pipeline {
    agent {
        node {
            label 'MLOpsA2'
        }
    }
    stages {
        stage('Checkout project') {
            steps {
                script {
                    git branch: "master",
                        credentialsId: 'adba6000-00d9-43dc-862e-1611c2e31519',
                        url: 'git@github.com:MahreenAthar/MLOps-Assignment-2.git'
                }
            }
        }
        stage('Installing packages') {
            steps {
                script {
                    sh 'pip install -r requirements.txt'
                }
            }
        }
        stage('Static Code Checking') {
            steps {
                script {
                    sh 'find . -name \\*.py | xargs pylint -f parseable | tee pylint.log'
                    recordIssues(
                        tool: pyLint(pattern: 'pylint.log'),
                        unstableTotalHigh: 100,
                    )
                }
            }
        }
    }
}