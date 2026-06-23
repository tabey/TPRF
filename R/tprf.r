library(R6)

TPRF <- R6Class(
  "TPRF",
  public = list(
    scaler_X_center = NULL,
    scaler_X_scale  = NULL,
    scaler_y_center = NULL,
    scaler_y_scale  = NULL,
    betas_ = NULL,

    fit = function(X, y) {
      X <- as.matrix(X)
      y <- as.numeric(y)

      # sklearn's StandardScaler uses population std (ddof = 0),
      # so we replicate that here rather than using R's scale() (ddof = 1).
      self$scaler_X_center <- colMeans(X)
      self$scaler_X_scale  <- apply(X, 2, function(v) sqrt(mean((v - mean(v))^2)))
      self$scaler_X_scale[self$scaler_X_scale == 0] <- 1

      self$scaler_y_center <- mean(y)
      self$scaler_y_scale  <- sqrt(mean((y - mean(y))^2))
      if (self$scaler_y_scale == 0) self$scaler_y_scale <- 1

      X_std <- sweep(sweep(X, 2, self$scaler_X_center, "-"),
                     2, self$scaler_X_scale, "/")
      y_std <- (y - self$scaler_y_center) / self$scaler_y_scale

      n_features <- ncol(X_std)
      self$betas_ <- numeric(n_features)

      for (i in seq_len(n_features)) {
        fit_i <- lm(y_std ~ X_std[, i])
        self$betas_[i] <- coef(fit_i)[2]
      }
      invisible(self)
    },

    transform = function(X) {
      if (is.null(self$betas_))
        stop("Must fit the model before transforming.")

      X <- as.matrix(X)
      X_std <- sweep(sweep(X, 2, self$scaler_X_center, "-"),
                     2, self$scaler_X_scale, "/")

      n_periods <- nrow(X_std)
      tprf_values <- numeric(n_periods)

      for (t in seq_len(n_periods)) {
        fit_t <- lm(X_std[t, ] ~ self$betas_)
        tprf_values[t] <- coef(fit_t)[2]
      }
      tprf_values
    },

    fit_transform = function(X, y) {
      self$fit(X, y)
      self$transform(X)
    }
  )
)