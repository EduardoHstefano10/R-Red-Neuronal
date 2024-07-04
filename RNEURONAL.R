# Cargar librerías necesarias
if (!require('randomForest')) install.packages('randomForest', dependencies=TRUE)
if (!require('neuralnet')) install.packages('neuralnet', dependencies=TRUE)
if (!require('caret')) install.packages('caret', dependencies=TRUE)

library(randomForest)
library(caret)
library(neuralnet)

set.seed(123) # Para reproducibilidad

# Definir control para la función RFE
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)

# Ejecutar RFE para seleccionar características
results <- rfe(CreditCard[,c("reports", "age", "income", "share", "expenditure", "dependents", "months", "majorcards", "active")],
               as.factor(CreditCard$card),
               sizes = c(1:9),
               rfeControl = control)

# Ver las características seleccionadas
print(results)
selected_features <- predictors(results)
print(selected_features)

# Dividir el conjunto de datos en entrenamiento (70%) y prueba (30%)
sample_index <- sample(seq_len(nrow(CreditCard)), size = 0.7 * nrow(CreditCard))
entrenamiento <- CreditCard[sample_index, ]
prueba <- CreditCard[-sample_index, ]

# Entrenar la red neuronal con las características seleccionadas
formula <- as.formula(paste("as.numeric(card == 'yes') ~", paste(selected_features, collapse = " + ")))
RNA1 <- neuralnet(formula, data = entrenamiento, hidden = 6)

# Plot de la red neuronal
plot(RNA1, rep = "best", main = "Red Neuronal con Características Seleccionadas")

# Predicciones en el conjunto de entrenamiento
output_train <- compute(RNA1, entrenamiento[,selected_features])
pred_train <- ifelse(output_train$net.result > 0.5, 'yes', 'no')
conf_matrix_train <- table(entrenamiento$card, pred_train)
print(conf_matrix_train)
TA1 <- sum(diag(conf_matrix_train)) / sum(conf_matrix_train)
cat("Tasa de aciertos en entrenamiento: ", round(TA1, 2), "\n")

# Plot de la matriz de confusión para entrenamiento
fourfoldplot(conf_matrix_train, color = c("#CC6666", "#99CC99"), conf.level = 0, main = "Matriz de Confusión (Entrenamiento)")

# Predicciones en el conjunto de prueba
output_test <- compute(RNA1, prueba[,selected_features])
pred_test <- ifelse(output_test$net.result > 0.5, 'yes', 'no')
conf_matrix_test <- table(prueba$card, pred_test)
print(conf_matrix_test)
TA2 <- sum(diag(conf_matrix_test)) / sum(conf_matrix_test)
cat("Tasa de aciertos en prueba: ", round(TA2, 2), "\n")

# Plot de la matriz de confusión para prueba
fourfoldplot(conf_matrix_test, color = c("#CC6666", "#99CC99"), conf.level = 0, main = "Matriz de Confusión (Prueba)")

