// JavaScript for Visa Processing Time Estimator

$(document).ready(function () {
    // Form submission handler
    $('#predictionForm').submit(function (e) {
        e.preventDefault();

        // Show loading spinner
        $('#loadingSpinner').show();
        $('#results').hide();

        // Disable submit button
        $('#predictBtn').prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Processing...');

        // Collect form data
        const formData = $(this).serialize();

        // Send AJAX request
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            success: function (response) {
                if (response.status === 'success') {
                    displayResults(response);
                } else {
                    showError(response.error || 'An error occurred');
                }
            },
            error: function (xhr, status, error) {
                showError('Server error: ' + error);
            },
            complete: function () {
                // Hide loading spinner
                $('#loadingSpinner').hide();

                // Re-enable submit button
                $('#predictBtn').prop('disabled', false).html('<i class="fas fa-calculator"></i> Estimate Processing Time');
            }
        });
    });

    // Display results function
    function displayResults(result) {
        const resultsHtml = `
            <div class="results-card card fade-in">
                <div class="card-header bg-success text-white">
                    <h4 class="mb-0"><i class="fas fa-check-circle"></i> Processing Time Estimate</h4>
                </div>
                <div class="card-body">
                    <!-- Main Result -->
                    <div class="text-center mb-4">
                        <div class="result-number">${result.processing_days}</div>
                        <div class="h4">Days</div>
                        <div class="confidence-interval">
                            95% Confidence: ${result.confidence_low} - ${result.confidence_high} days
                        </div>
                    </div>
                    
                    <!-- Alternative Timeframes -->
                    <div class="row text-center mb-4">
                        <div class="col-md-4">
                            <div class="h5">${result.processing_weeks}</div>
                            <div class="text-muted">Weeks</div>
                        </div>
                        <div class="col-md-4">
                            <div class="h5">${result.processing_months}</div>
                            <div class="text-muted">Months</div>
                        </div>
                        <div class="col-md-4">
                            <div class="speed-indicator speed-${result.speed_color}">
                                ${result.speed_category} Processing
                            </div>
                        </div>
                    </div>
                    
                    <!-- Application Summary -->
                    <div class="card mb-3">
                        <div class="card-header">
                            <h6 class="mb-0"><i class="fas fa-file-alt"></i> Application Summary</h6>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <p><strong>Visa Class:</strong> ${result.input_summary.visa_class}</p>
                                    <p><strong>Employer State:</strong> ${result.input_summary.employer_state}</p>
                                    <p><strong>Worksite State:</strong> ${result.input_summary.worksite_state}</p>
                                    <p><strong>Job Title:</strong> ${result.input_summary.job_title}</p>
                                    <p><strong>SOC Title:</strong> ${result.input_summary.soc_title}</p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>Wage:</strong> $${result.input_summary.wage_from} / ${result.input_summary.wage_unit}</p>
                                    <p><strong>NAICS Code:</strong> ${result.input_summary.naics_code}</p>
                                    <p><strong>Full Time:</strong> ${result.input_summary.full_time === 'Y' ? 'Yes' : 'No'}</p>
                                    <p><strong>H-1B Dependent:</strong> ${result.input_summary.h1b_dependent === 'Y' ? 'Yes' : (result.input_summary.h1b_dependent === 'N' ? 'No' : result.input_summary.h1b_dependent)}</p>
                                    <p><strong>Willful Violator:</strong> ${result.input_summary.willful_violator === 'Y' ? 'Yes' : (result.input_summary.willful_violator === 'N' ? 'No' : result.input_summary.willful_violator)}</p>
                                </div>
                            </div>
                            <hr>
                            <div class="row">
                                <div class="col-md-6">
                                    <p class="mb-0"><strong>Season:</strong> ${result.input_summary.season}</p>
                                </div>
                                <div class="col-md-6">
                                    <p class="mb-0"><strong>Submission Date:</strong> ${result.input_summary.submission_date}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Next Steps -->
                    <div class="alert alert-info">
                        <h6><i class="fas fa-lightbulb"></i> Suggestions:</h6>
                        <ul class="mb-0">
                            <li>Submit your application as early as possible</li>
                            <li>Ensure all documents are complete and accurate</li>
                            <li>Monitor your application status regularly</li>
                            <li>Consider premium processing if available for your visa type</li>
                        </ul>
                    </div>
                    
                    <!-- Print Button -->
                    <div class="d-grid gap-2 d-md-flex justify-content-md-end">
                        <button class="btn btn-outline-primary no-print" onclick="window.print()">
                            <i class="fas fa-print"></i> Print Results
                        </button>
                        <button class="btn btn-primary no-print" onclick="resetForm()">
                            <i class="fas fa-redo"></i> New Estimate
                        </button>
                    </div>
                </div>
            </div>
        `;

        $('#results').html(resultsHtml).show();

        // Scroll to results
        $('html, body').animate({
            scrollTop: $('#results').offset().top - 100
        }, 500);
    }

    // Show error function
    function showError(message) {
        const errorHtml = `
            <div class="alert alert-danger fade-in" role="alert">
                <h5><i class="fas fa-exclamation-triangle"></i> Error</h5>
                <p class="mb-0">${message}</p>
            </div>
        `;
        $('#results').html(errorHtml).show();
    }

    // Reset form function
    window.resetForm = function () {
        $('#predictionForm')[0].reset();
        $('#results').hide().empty();
        $('html, body').animate({
            scrollTop: 0
        }, 500);
    };

    // Auto-fill SOC title based on job title
    $('#job_title').on('blur', function () {
        const jobTitle = $(this).val().toLowerCase();
        if (jobTitle && !$('#soc_title').val()) {
            // Simple mapping (in production, use a proper API or database)
            const socMapping = {
                'software': 'Software Developers, Applications',
                'developer': 'Software Developers, Applications',
                'engineer': 'Software Engineers',
                'analyst': 'Computer Systems Analysts',
                'manager': 'Computer and Information Systems Managers',
                'scientist': 'Data Scientists',
                'designer': 'Web Developers',
                'consultant': 'Management Analysts',
                'accountant': 'Accountants',
                'teacher': 'Secondary School Teachers'
            };

            for (const [keyword, soc] of Object.entries(socMapping)) {
                if (jobTitle.includes(keyword)) {
                    $('#soc_title').val(soc);
                    break;
                }
            }
        }
    });

    // Set default prevailing wage to 5% above wage
    $('#wage_from').on('blur', function () {
        const wage = parseFloat($(this).val());
        if (wage && !$('#prevailing_wage').val()) {
            const prevailingWage = Math.round(wage * 1.05);
            $('#prevailing_wage').val(prevailingWage);
        }
    });

    // Copy employer state to worksite state if empty
    $('#employer_state').on('change', function () {
        if (!$('#worksite_state').val()) {
            $('#worksite_state').val($(this).val());
        }
    });
});